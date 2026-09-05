import asyncio
import json
import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from app.agents.planner import PlannerAgent
from app.api.dependencies import get_current_user, require_workspace_role
from app.api.deps_providers import get_conversation_repository, get_planner_agent, get_redis_cache
from app.api.rate_limit_dependencies import rate_limit_by_user
from app.api.schemas import (
    AgentJobResponse,
    AgentJobStatusResponse,
    ConversationCreateRequest,
    ConversationResponse,
    ConversationUpdateRequest,
    MessageCreateRequest,
    MessageResponse,
)
from app.cache.redis_client import RedisCache
from app.conversations.repository import ConversationRepository
from app.conversations.summarization import build_effective_history
from app.core.config import settings
from app.domain.entities import Conversation, Message, User
from app.jobs.queue import build_job_queue, JobQueueError


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _send_inngest_event(name: str, data: dict) -> bool:
    if settings.environment == "test":
        return False
    try:
        import inngest
        from app.inngest.client import inngest_client

        inngest_client.send_sync(inngest.Event(name=name, data=data))
        return True
    except Exception as exc:
        logger.info("Inngest dispatch failed for %s; falling back when possible: %s", name, exc)
        return False


def _job_result_message_response(result: dict) -> MessageResponse:
    return MessageResponse(
        id=result["id"],
        role=result["role"],
        content=result["content"],
        tool_name=result.get("tool_name"),
        tool_output=result.get("tool_output"),
        agent_name=result.get("agent_name"),
        tool_arguments=result.get("tool_arguments"),
        thought_process=result.get("thought_process"),
        created_at=result["created_at"],
    )


@router.get("", response_model=list[ConversationResponse])
def list_conversations(
    current_user: Annotated[User, Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    resolved_ws_id: Annotated[str, Depends(require_workspace_role("owner", "member", "viewer"))],
    workspace_id: str | None = None,
) -> list[ConversationResponse]:
    conversations = repo.list_for_workspace(resolved_ws_id)
    return [_conversation_response(conversation) for conversation in conversations]


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    resolved_ws_id: Annotated[str, Depends(require_workspace_role("owner", "member"))],
    workspace_id: str | None = None,
) -> ConversationResponse:
    conversation = repo.create(user_id=current_user.id, title=payload.title.strip(), workspace_id=resolved_ws_id)
    return _conversation_response(conversation)


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
def list_messages(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    _: Annotated[str, Depends(require_workspace_role("owner", "member", "viewer"))],
) -> list[MessageResponse]:
    _require_conversation(repo, conversation_id)
    return [_message_response(message) for message in repo.list_messages(conversation_id)]


@router.post(
    "/{conversation_id}/messages",
    response_model=AgentJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit_by_user("chat_message", limit=20, window_seconds=60))],
)
def send_message(
    conversation_id: str,
    payload: MessageCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    _: Annotated[str, Depends(require_workspace_role("owner", "member"))],
    cache: Annotated[RedisCache | None, Depends(get_redis_cache)] = None,
) -> AgentJobResponse:
    _require_conversation(repo, conversation_id)

    content = payload.content.strip()
    images = [image.model_dump() for image in payload.images]
    stored_content = content if content else "[Image attached]"
    user_message = repo.add_message(
        conversation_id=conversation_id,
        role="user",
        content=stored_content,
        user_id=current_user.id,
    )

    chat_payload = {
        "conversation_id": conversation_id,
        "user_id": current_user.id,
        "user_message_id": user_message.id,
        "content": content or "Please analyze the attached image(s).",
        "images": images,
    }

    queue = build_job_queue()
    if queue is not None:
        lock_key = f"active_chat_job:{conversation_id}"
        lock_claimed = False
        if cache:
            try:
                lock_claimed = cache.set_if_not_exists(lock_key, "pending", ttl_seconds=300)
                if not lock_claimed:
                    existing_job_id = cache.get(lock_key)
                    job = queue.get(existing_job_id) if existing_job_id else None
                    if job is None or job.status.value not in ["queued", "running"]:
                        cache.delete(lock_key)
                        lock_claimed = cache.set_if_not_exists(lock_key, "pending", ttl_seconds=300)
                    if not lock_claimed:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail="Archimedes is currently answering a query in this conversation. Please wait until it completes.",
                        )
            except JobQueueError as exc:
                logger.error("Error checking existing job status: %s", exc)

        job_id = str(uuid4())
        chat_payload = {**chat_payload, "job_id": job_id}
        try:
            job = queue.create_job("chat_agent_run", chat_payload, job_id=job_id)
            dispatched_to_inngest = _send_inngest_event("ai-os/chat.requested", chat_payload)
            if not dispatched_to_inngest:
                queue.enqueue_existing(job.id)
            if cache:
                cache.set(lock_key, job.id, ttl_seconds=300)
            return AgentJobResponse(job_id=job.id, status=job.status.value, user_message=_message_response(user_message))
        except Exception as exc:
            if cache and lock_claimed:
                cache.delete(lock_key)
            logger.warning("Queue enqueue failed (%s), falling back to synchronous execution", exc)

    # 3. Synchronous fallback execution if queue is disabled/failing
    from app.jobs.chat_agent import run_chat_agent_job
    fallback_job_id = f"sync_{uuid4().hex[:12]}"
    try:
        result = run_chat_agent_job({
            **chat_payload,
            "job_id": fallback_job_id,
        })
    except Exception as exc:
        logger.exception("Synchronous fallback chat agent run failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to process message: {exc}",
        ) from exc

    return AgentJobResponse(
        job_id=fallback_job_id,
        status="succeeded",
        user_message=_message_response(user_message),
        assistant_message=_job_result_message_response(result),
    )


# EXPERIMENTAL/UNUSED-BY-DEFAULT: Synchronous SSE chat endpoint.
# Reintroducing this as default can block the FastAPI event loop for concurrent users.
@router.post(
    "/{conversation_id}/messages/stream",
    dependencies=[Depends(rate_limit_by_user("chat_message_stream", limit=30, window_seconds=60))],
)
async def stream_message(
    conversation_id: str,
    payload: MessageCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    agent: Annotated[PlannerAgent, Depends(get_planner_agent)],
    _: Annotated[str, Depends(require_workspace_role("owner", "member"))],
    cache: Annotated[RedisCache | None, Depends(get_redis_cache)] = None,
) -> StreamingResponse:
    _require_conversation(repo, conversation_id)

    content = payload.content.strip()
    user_message = repo.add_message(
        conversation_id=conversation_id,
        role="user",
        content=content,
        user_id=current_user.id,
    )

    async def event_generator():
        all_messages = [
            m for m in repo.list_messages(conversation_id)
            if m.id != user_message.id
        ]
        history = build_effective_history(
            messages=all_messages,
            llm_provider=getattr(agent, "llm_provider", None),
            cache=cache,
            conversation_id=conversation_id,
        )

        yield f"event: thinking\ndata: {json.dumps({'status': 'Planner analyzing prompt...'})}\n\n"
        await asyncio.sleep(0.01)

        try:
            result = await asyncio.to_thread(agent.run, user_input=content, history=history)

            if result.thought_process:
                yield f"event: thought\ndata: {json.dumps({'thought': result.thought_process})}\n\n"
                await asyncio.sleep(0.01)

            if result.agent_name:
                yield f"event: agent_route\ndata: {json.dumps({'agent_name': result.agent_name})}\n\n"
                await asyncio.sleep(0.01)

            if result.tool_name:
                tool_args = result.tool_arguments if isinstance(getattr(result, "tool_arguments", None), dict) else {}
                yield f"event: tool_start\ndata: {json.dumps({'tool_name': result.tool_name, 'tool_arguments': tool_args})}\n\n"
                await asyncio.sleep(0.01)

            answer = result.answer
            chunk_size = 16
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"event: token\ndata: {json.dumps({'delta': chunk})}\n\n"
                await asyncio.sleep(0.01)

            assistant_message = repo.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                tool_name=result.tool_name,
            )
            yield f"event: done\ndata: {json.dumps({'message_id': assistant_message.id, 'role': 'assistant', 'content': answer, 'tool_name': result.tool_name})}\n\n"

        except Exception as exc:
            logger.error("Error during streaming chat execution: %s", exc, exc_info=True)
            error_text = f"Request failed: {exc}"
            repo.add_message(conversation_id=conversation_id, role="assistant", content=error_text)
            yield f"event: error\ndata: {json.dumps({'error': error_text})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{conversation_id}/messages/jobs/{job_id}", response_model=AgentJobStatusResponse)
def get_message_job(
    conversation_id: str,
    job_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    _: Annotated[str, Depends(require_workspace_role("owner", "member", "viewer"))],
) -> AgentJobStatusResponse:
    # Synchronous fallback jobs already completed — return result from DB
    if job_id.startswith("sync_"):
        messages = repo.list_messages(conversation_id)
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assistant_message = None
        if assistant_msgs:
            latest = assistant_msgs[-1]
            assistant_message = MessageResponse(
                id=latest.id,
                role=latest.role,
                content=latest.content,
                tool_name=getattr(latest, "tool_name", None),
                tool_output=getattr(latest, "tool_output", None),
                agent_name=getattr(latest, "agent_name", None),
                tool_arguments=getattr(latest, "tool_arguments", None),
                thought_process=getattr(latest, "thought_process", None),
                created_at=latest.created_at.isoformat() if hasattr(latest.created_at, "isoformat") else str(latest.created_at),
            )
        return AgentJobStatusResponse(
            job_id=job_id,
            status="succeeded",
            assistant_message=assistant_message,
            execution_steps=None,
            error=None,
        )

    queue = build_job_queue()
    if queue is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Job queue unavailable.")
    try:
        job = queue.get(job_id)
    except JobQueueError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    if (
        job is None
        or job.payload.get("conversation_id") != conversation_id
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    assistant_message = None
    if job.result is not None:
        assistant_message = _job_result_message_response(job.result)
    return AgentJobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        assistant_message=assistant_message,
        execution_steps=job.execution_steps,
        error=job.error,
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    _: Annotated[str, Depends(require_workspace_role("owner", "member"))],
) -> Response:
    deleted = repo.delete(conversation_id=conversation_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: str,
    payload: ConversationUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
    _: Annotated[str, Depends(require_workspace_role("owner", "member"))],
) -> ConversationResponse:
    conversation = repo.update_title(conversation_id=conversation_id, title=payload.title.strip())
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return _conversation_response(conversation)


def _require_conversation(repo: ConversationRepository, conversation_id: str) -> Conversation:
    conversation = repo.get_by_id(conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


def _conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _message_response(message: Message) -> MessageResponse:
    import json
    tool_args = None
    if message.tool_arguments:
        try:
            tool_args = json.loads(message.tool_arguments)
        except Exception:
            tool_args = {"raw": message.tool_arguments}
    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        tool_name=message.tool_name,
        tool_output=message.tool_output,
        agent_name=message.agent_name,
        tool_arguments=tool_args,
        created_at=message.created_at,
        user_id=message.user_id,
        user_name=message.user_name,
        user_email=message.user_email,
    )

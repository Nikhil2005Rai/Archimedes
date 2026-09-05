import json
import logging

from app.agents.graph import MultiAgentGraph
from app.agents.planner import PlannerResult
from app.agents.registry import build_agent_registry
from app.auth.provider_resolution import ResolvedProviderConfig, resolve_active_provider_config, resolve_gemini_api_key
from app.cache.redis_client import build_redis_cache
from app.conversations.caching import CachingConversationRepository
from app.conversations.repository import ConversationRepository
from app.core.config import settings
from app.db import SessionLocal, get_engine
from app.jobs.queue import build_job_queue
from app.infrastructure.models import UserModel
from app.providers.base import LLMGenerationError, LLMImage, LLMMessage
from app.providers.caching import CachingLLMProvider
from app.providers.embeddings.errors import EmbeddingError
from app.providers.embeddings.gemini import GeminiEmbeddingProvider
from app.providers.registry import build_provider
from app.conversations.summarization import build_effective_history
from app.retrieval.repository import RetrievalRepository
from app.tools.registry import build_tool_registry
from app.tools.repository import ToolCallRepository

logger = logging.getLogger(__name__)


def _add_failure_message(conversation_id: str, content: str) -> None:
    session = SessionLocal()
    try:
        ConversationRepository(session).add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
        )
    except Exception:
        logger.exception("Failed to persist assistant failure message")
    finally:
        session.close()


def run_chat_agent_job(payload: dict) -> dict:
    """payload: {"conversation_id": str, "user_id": str, "user_message_id": str, "content": str, "images"?: list}"""
    get_engine()

    conversation_id = payload["conversation_id"]
    user_id = payload["user_id"]
    user_message_id = payload["user_message_id"]
    content = payload["content"]
    raw_images = payload.get("images") or []
    images = [
        LLMImage(mime_type=image["mime_type"], data=image["data"])
        for image in raw_images
    ]
    has_images = len(images) > 0
    job_id = payload.get("job_id")

    cache = build_redis_cache()
    queue = build_job_queue()

    # Phase 1: read all DB-backed inputs, then release the connection before any LLM call.
    session = SessionLocal()
    user_db_url = None
    user_db_session = None
    user_engine = None
    try:
        user_model = session.get(UserModel, user_id)
        if user_model is None:
            raise ValueError("User not found")
        preferred_provider = user_model.preferred_provider
        preferred_model = user_model.preferred_model

        inner_repo = ConversationRepository(session)
        conv = inner_repo.get_by_id(conversation_id)
        workspace_id = conv.workspace_id if conv else None
        all_messages = [
            m for m in inner_repo.list_messages(conversation_id)
            if m.id != user_message_id
        ]

        try:
            gemini_key = resolve_gemini_api_key(session, user_id, preferred_provider)
        except ValueError as exc:
            message = "Image input requires a Gemini API key; save one in BYOK settings." if has_images else str(exc)
            _add_failure_message(conversation_id, f"Request failed: {message}")
            raise ValueError(message) from exc

        if has_images:
            provider_config = ResolvedProviderConfig(provider_name="gemini", api_key=gemini_key)
            preferred_model = preferred_model if preferred_provider == "gemini" else None
        else:
            try:
                provider_config = resolve_active_provider_config(session, user_id, preferred_provider)
            except ValueError as exc:
                _add_failure_message(conversation_id, f"Request failed: LLM provider failed: {exc}")
                raise ValueError(f"LLM provider failed: {exc}") from exc

        from app.auth.api_key_repository import UserApiKeyRepository
        db_key = UserApiKeyRepository(session).get_for_user_provider(user_id, "database")
        if db_key is not None:
            try:
                from app.auth.encryption import EncryptionService
                user_db_url = EncryptionService().decrypt(db_key.encrypted_key)
            except Exception:
                logger.exception("Failed to decrypt user's database URL")
    finally:
        session.close()

    # Phase 2: build providers and perform slow model work without the Phase 1 session open.
    llm_provider = build_provider(
        api_key=provider_config.api_key,
        provider_name=provider_config.provider_name,
        model=preferred_model,
        base_url=provider_config.base_url,
    )
    if cache is not None:
        llm_provider = CachingLLMProvider(inner=llm_provider, cache=cache, user_id=user_id)

    embedding_provider = GeminiEmbeddingProvider(
        api_key=gemini_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    history = build_effective_history(
        messages=all_messages,
        llm_provider=llm_provider,
        cache=cache,
        conversation_id=conversation_id,
    )

    if has_images:
        if queue and job_id:
            queue.add_execution_step(job_id, "vision", "Gemini analyzing image input...", "running")
        try:
            vision_messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "You are Archimedes' Gemini vision path. Analyze the attached image(s) carefully "
                        "and answer the user's request directly. If the user asks about visible text, "
                        "transcribe it only as needed for the answer."
                    ),
                )
            ]
            if history:
                vision_messages.extend(history[-12:])
            vision_messages.append(LLMMessage(role="user", content=content, images=images))
            response = llm_provider.generate(vision_messages)
            result = PlannerResult(
                answer=response.content,
                agent_name="gemini",
                thought_process=response.thought,
            )
            if queue and job_id:
                if result.thought_process:
                    queue.add_execution_step(
                        job_id,
                        "thinking",
                        "Model Reasoning",
                        "completed",
                        metadata={"thought": result.thought_process},
                    )
                queue.add_execution_step(
                    job_id,
                    "specialist",
                    "Routed to Gemini Vision",
                    "completed",
                    metadata={"agent_name": "gemini"},
                )
                queue.add_execution_step(job_id, "finalize", "Finalized response", "completed")
        except LLMGenerationError as exc:
            if queue and job_id:
                queue.add_execution_step(job_id, "error", f"LLM provider failed: {exc}", "failed")
            _add_failure_message(conversation_id, f"Request failed: LLM provider failed: {exc}")
            raise ValueError(f"LLM provider failed: {exc}") from exc
        except Exception as exc:
            if queue and job_id:
                queue.add_execution_step(job_id, "error", "An unexpected server error occurred.", "failed")
            logger.exception("Unexpected error during Gemini vision execution")
            _add_failure_message(conversation_id, "Request failed: An unexpected server error occurred.")
            raise ValueError("An unexpected server error occurred.") from exc
    else:
        agent_session = SessionLocal()
        try:
            if user_db_url:
                try:
                    from sqlalchemy import create_engine
                    from sqlalchemy.orm import sessionmaker
                    user_engine = create_engine(user_db_url)
                    SessionLocalUser = sessionmaker(bind=user_engine)
                    user_db_session = SessionLocalUser()
                except Exception:
                    logger.exception("Failed to connect to user's database")

            agent = MultiAgentGraph(
                llm_provider=llm_provider,
                tools=build_tool_registry(agent_session, user_db_session=user_db_session),
                agents=build_agent_registry(),
                embedding_provider=embedding_provider,
                retrieval_repository=RetrievalRepository(agent_session),
                user_id=user_id,
                workspace_id=workspace_id,
            )

            if queue and job_id:
                queue.add_execution_step(job_id, "planner", "Planner analyzing prompt...", "running")

            try:
                result = agent.run(user_input=content, history=history)
                if queue and job_id:
                    if result.thought_process:
                        queue.add_execution_step(
                            job_id,
                            "thinking",
                            "Model Reasoning",
                            "completed",
                            metadata={"thought": result.thought_process},
                        )
                    if result.agent_name:
                        queue.add_execution_step(
                            job_id,
                            "specialist",
                            f"Routed to {result.agent_name.capitalize()} Agent",
                            "completed",
                            metadata={"agent_name": result.agent_name},
                        )
                    if result.tool_name:
                        queue.add_execution_step(
                            job_id,
                            "tool",
                            f"Executed tool `{result.tool_name}`",
                            "completed",
                            metadata={
                                "tool_name": result.tool_name,
                                "tool_arguments": result.tool_arguments,
                                "tool_output": result.tool_output[:200] if result.tool_output else None,
                            },
                        )
                    queue.add_execution_step(job_id, "finalize", "Finalized response", "completed")
            except LLMGenerationError as exc:
                if queue and job_id:
                    queue.add_execution_step(job_id, "error", f"LLM provider failed: {exc}", "failed")
                _add_failure_message(conversation_id, f"Request failed: LLM provider failed: {exc}")
                raise ValueError(f"LLM provider failed: {exc}") from exc
            except EmbeddingError as exc:
                if queue and job_id:
                    queue.add_execution_step(job_id, "error", f"Embedding provider failed: {exc}", "failed")
                _add_failure_message(conversation_id, f"Request failed: Embedding provider failed: {exc}")
                raise ValueError(f"Embedding provider failed: {exc}") from exc
            except Exception as exc:
                if queue and job_id:
                    queue.add_execution_step(job_id, "error", "An unexpected server error occurred.", "failed")
                logger.exception("Unexpected error during agent execution")
                _add_failure_message(conversation_id, "Request failed: An unexpected server error occurred.")
                raise ValueError("An unexpected server error occurred.") from exc
        finally:
            agent_session.close()
            if user_db_session is not None:
                user_db_session.close()
            if user_engine is not None:
                user_engine.dispose()

    # Phase 3: persist results with a fresh session/connection.
    write_session = SessionLocal()
    try:
        inner_repo = ConversationRepository(write_session)
        repo = CachingConversationRepository(inner=inner_repo, cache=cache) if cache else inner_repo
        assistant_message = repo.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result.answer,
            tool_name=result.tool_name,
        )
        if result.tool_name and result.tool_arguments is not None and result.tool_output is not None:
            ToolCallRepository(write_session).create(
                conversation_id=conversation_id,
                message_id=assistant_message.id,
                tool_name=result.tool_name,
                agent_name=result.agent_name or "planner",
                arguments=json.dumps(result.tool_arguments),
                output=result.tool_output,
            )
        if result.retrieval_query is not None:
            RetrievalRepository(write_session).create(
                conversation_id=conversation_id,
                message_id=assistant_message.id,
                agent_name=result.agent_name or "knowledge",
                query=result.retrieval_query,
                chunk_ids=result.retrieval_chunk_ids or [],
                scores=result.retrieval_scores or [],
            )

        return {
            "id": assistant_message.id,
            "role": assistant_message.role,
            "content": assistant_message.content,
            "tool_name": assistant_message.tool_name,
            "tool_output": assistant_message.tool_output,
            "created_at": assistant_message.created_at.isoformat(),
            "agent_name": result.agent_name,
            "tool_arguments": result.tool_arguments,
            "thought_process": result.thought_process,
        }
    finally:
        write_session.close()

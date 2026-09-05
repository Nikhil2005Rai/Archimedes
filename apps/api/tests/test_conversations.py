from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps_providers import get_embedding_provider, get_llm_provider, get_retrieval_repository
from app.domain.entities import RetrievedChunk
from app.infrastructure.models import ToolCallModel
from app.main import app
from app.providers.base import LLMGenerationError, LLMMessage, LLMResponse, LLMToolCall, ToolSchema
from app.jobs.entities import JobStatus


class FakeUpstashRedisClient:
    def __init__(self) -> None:
        self.db: dict[str, str] = {}
        self.queues: dict[str, list[str]] = {}

    def get(self, key: str) -> str | None:
        return self.db.get(key)

    def set(self, key: str, value: str, nx: bool | None = None, ex: int | None = None, **kwargs):
        if nx and key in self.db:
            return None
        self.db[key] = value
        return True

    def rpush(self, key: str, value: str) -> int:
        if key not in self.queues:
            self.queues[key] = []
        self.queues[key].append(value)
        return len(self.queues[key])

    def lpop(self, key: str) -> str | None:
        if key not in self.queues or not self.queues[key]:
            return None
        return self.queues[key].pop(0)


class ToolCallingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[LLMMessage], tools: list[ToolSchema] | None = None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="ROUTE: research")
        if self.calls == 2:
            return LLMResponse(content="", tool_call=LLMToolCall(name="current_time", arguments={}))
        return LLMResponse(content="The current time has been checked by research.")


class KnowledgeRoutingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[LLMMessage], tools: list[ToolSchema] | None = None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="ROUTE: knowledge")
        context = messages[0].content
        if "alpha beta gamma" in context:
            return LLMResponse(content="Grounded answer from uploaded notes.")
        return LLMResponse(content="No grounded context found.")


class FailingEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        from app.providers.embeddings.errors import EmbeddingError
        raise EmbeddingError("Gemini returned 3072 embedding dimensions, expected 768.")


class FakeEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


class FakeRetrievalRepository:
    def search(
        self,
        user_id: str,
        embedding: list[float],
        limit: int = 4,
        workspace_id: str | None = None,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                id="chunk-1",
                document_id="document-1",
                content="alpha beta gamma",
                score=0.01,
            )
        ]

    def create(self, *args, **kwargs) -> None:
        pass


def test_create_list_send_message_and_persist_assistant_reply(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    from app.jobs.queue import JobQueue
    from app.jobs.chat_agent import run_chat_agent_job
    from app.providers.base import LLMResponse

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = FakeUpstashRedisClient()
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Planner session"})
    assert create.status_code == 201
    conversation_id = create.json()["id"]

    conversations = client.get("/conversations", headers=auth_headers)
    assert conversations.status_code == 200
    assert conversations.json()[0]["title"] == "Planner session"

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "Hello planner"},
    )
    assert send.status_code == 202
    job_id = send.json()["job_id"]
    assert job_id is not None
    assert send.json()["user_message"]["content"] == "Hello planner"

    status_response = client.get(f"/conversations/{conversation_id}/messages/jobs/{job_id}", headers=auth_headers)
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "queued"
    assert status_response.json()["assistant_message"] is None

    class FakeAssistantProvider:
        def generate(self, messages, tools=None):
            return LLMResponse(content="Fake assistant reply")

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: FakeAssistantProvider())
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    job = fake_queue.dequeue("chat_agent_run")
    assert job is not None
    assert job.id == job_id

    result = run_chat_agent_job(job.payload)
    fake_queue.update_status(job.id, JobStatus.SUCCEEDED, result=result)

    status_response = client.get(f"/conversations/{conversation_id}/messages/jobs/{job_id}", headers=auth_headers)
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "succeeded"
    assert status_response.json()["assistant_message"]["content"] == "Fake assistant reply"

    messages = client.get(f"/conversations/{conversation_id}/messages", headers=auth_headers)
    assert messages.status_code == 200
    assert [message["role"] for message in messages.json()] == ["user", "assistant"]
    assert messages.json()[1]["content"] == "Fake assistant reply"


def test_send_message_accepts_image_only_payload_and_rejects_more_than_two_images(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.jobs.queue import JobQueue

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = FakeUpstashRedisClient()
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Vision session"})
    assert create.status_code == 201
    conversation_id = create.json()["id"]

    image = {"mime_type": "image/png", "data": "aGVsbG8="}
    too_many = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "Describe these", "images": [image, image, image]},
    )
    assert too_many.status_code == 422

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"images": [image]},
    )
    assert send.status_code == 202
    assert send.json()["user_message"]["content"] == "[Image attached]"

    job = fake_queue.dequeue("chat_agent_run")
    assert job is not None
    assert job.payload["content"] == "Please analyze the attached image(s)."
    assert job.payload["images"] == [image]


def test_send_message_logs_tool_call_when_provider_requests_tool(
    client: TestClient,
    auth_headers: dict[str, str],
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.jobs.queue import JobQueue
    from app.jobs.chat_agent import run_chat_agent_job

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = FakeUpstashRedisClient()
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Tool session"})
    conversation_id = create.json()["id"]

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "What time is it?"},
    )
    assert send.status_code == 202
    job_id = send.json()["job_id"]

    provider = ToolCallingProvider()
    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: provider)
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    job = fake_queue.dequeue("chat_agent_run")
    assert job is not None
    result = run_chat_agent_job(job.payload)
    fake_queue.update_status(job.id, JobStatus.SUCCEEDED, result=result)

    tool_call = db_session.scalar(select(ToolCallModel).where(ToolCallModel.tool_name == "current_time"))
    assert tool_call is not None
    assert tool_call.conversation_id == conversation_id
    assert tool_call.message_id == result["id"]
    assert tool_call.agent_name == "research"
    assert tool_call.arguments == "{}"
    assert "Current local time is" in tool_call.output

    status_response = client.get(f"/conversations/{conversation_id}/messages/jobs/{job_id}", headers=auth_headers)
    assert status_response.status_code == 200
    res_data = status_response.json()
    assert res_data["status"] == "succeeded"
    assert res_data["assistant_message"]["tool_name"] == "current_time"
    assert res_data["assistant_message"]["agent_name"] == "research"
    assert res_data["assistant_message"]["tool_arguments"] == {}


def test_send_message_returns_502_when_knowledge_embedding_fails(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    from app.jobs.queue import JobQueue
    from app.jobs.chat_agent import run_chat_agent_job

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = FakeUpstashRedisClient()
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Knowledge session"})
    conversation_id = create.json()["id"]

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "Using my uploaded knowledge, answer this."},
    )
    assert send.status_code == 202
    job_id = send.json()["job_id"]

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: KnowledgeRoutingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FailingEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    job = fake_queue.dequeue("chat_agent_run")
    assert job is not None

    with pytest.raises(ValueError) as excinfo:
        run_chat_agent_job(job.payload)
    
    assert "Embedding provider failed" in str(excinfo.value)
    assert "expected 768" in str(excinfo.value)


def test_upload_document_then_knowledge_question_returns_grounded_answer(
    client: TestClient,
    auth_headers: dict[str, str],
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infrastructure.models import UserModel
    from app.retrieval.repository import DocumentRepository
    from app.jobs.queue import JobQueue
    from app.jobs.chat_agent import run_chat_agent_job

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = FakeUpstashRedisClient()
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    user = db_session.scalar(select(UserModel).where(UserModel.email == "user@example.com"))
    assert user is not None

    DocumentRepository(db_session).create_with_chunks(
        user_id=user.id,
        title="Notes",
        source_type="pasted_text",
        chunks=["alpha beta gamma"],
        embeddings=[[0.1] * 768],
    )

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: KnowledgeRoutingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.RetrievalRepository", lambda sess: FakeRetrievalRepository())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Knowledge session"})
    conversation_id = create.json()["id"]

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "Using my uploaded knowledge, answer this."},
    )
    assert send.status_code == 202
    job_id = send.json()["job_id"]

    job = fake_queue.dequeue("chat_agent_run")
    assert job is not None
    result = run_chat_agent_job(job.payload)
    fake_queue.update_status(job.id, JobStatus.SUCCEEDED, result=result)

    status_response = client.get(f"/conversations/{conversation_id}/messages/jobs/{job_id}", headers=auth_headers)
    assert status_response.status_code == 200
    assert status_response.json()["assistant_message"]["content"] == "Grounded answer from uploaded notes."


class GenerationFailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[LLMMessage], tools: list[ToolSchema] | None = None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="ROUTE: research")
        raise LLMGenerationError("Gemini stopped with finishReason=MAX_TOKENS before producing an answer.")


def test_send_message_returns_502_when_generation_fails(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    from app.jobs.queue import JobQueue
    from app.jobs.chat_agent import run_chat_agent_job

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = FakeUpstashRedisClient()
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Research session"})
    conversation_id = create.json()["id"]

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "What is the time?"},
    )
    assert send.status_code == 202
    job_id = send.json()["job_id"]

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: GenerationFailingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    job = fake_queue.dequeue("chat_agent_run")
    assert job is not None

    with pytest.raises(ValueError) as excinfo:
        run_chat_agent_job(job.payload)
    
    assert "LLM provider failed" in str(excinfo.value)
    assert "MAX_TOKENS" in str(excinfo.value)


def test_delete_conversation_cascade_and_security(
    client: TestClient,
    auth_headers: dict[str, str],
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    create_auth_headers,
) -> None:
    from app.infrastructure.models import ConversationModel, MessageModel
    from app.jobs.queue import JobQueue

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = FakeUpstashRedisClient()
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Session to Delete"})
    assert create.status_code == 201
    conversation_id = create.json()["id"]

    user_msg = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "trigger tool call"},
    )
    assert user_msg.status_code == 202
    
    messages_before = db_session.scalars(select(MessageModel).where(MessageModel.conversation_id == conversation_id)).all()
    assert len(messages_before) > 0
    
    other_headers = create_auth_headers("other@example.com")
    
    bad_delete = client.delete(f"/conversations/{conversation_id}", headers=other_headers)
    assert bad_delete.status_code == 404
    
    good_delete = client.delete(f"/conversations/{conversation_id}", headers=auth_headers)
    assert good_delete.status_code == 204
    
    deleted_conv = db_session.get(ConversationModel, conversation_id)
    assert deleted_conv is None
    
    messages_after = db_session.scalars(select(MessageModel).where(MessageModel.conversation_id == conversation_id)).all()
    assert len(messages_after) == 0


def test_rename_conversation(
    client: TestClient,
    auth_headers: dict[str, str],
    db_session: Session,
    create_auth_headers,
) -> None:
    create = client.post("/conversations", headers=auth_headers, json={"title": "Old Name"})
    assert create.status_code == 201
    conversation_id = create.json()["id"]
    
    other_headers = create_auth_headers("other2@example.com")
    
    bad_rename = client.put(f"/conversations/{conversation_id}", headers=other_headers, json={"title": "Hacked Title"})
    assert bad_rename.status_code == 404
    
    invalid_rename = client.put(f"/conversations/{conversation_id}", headers=auth_headers, json={"title": ""})
    assert invalid_rename.status_code == 422
    
    good_rename = client.put(f"/conversations/{conversation_id}", headers=auth_headers, json={"title": "New Title"})
    assert good_rename.status_code == 200
    assert good_rename.json()["title"] == "New Title"
    
    from app.infrastructure.models import ConversationModel
    db_conv = db_session.get(ConversationModel, conversation_id)
    assert db_conv is not None
    assert db_conv.title == "New Title"


class GenericErrorLLMProvider:
    def generate(self, messages, tools=None):
        raise RuntimeError("Something went wrong inside the model call")


def test_send_message_returns_500_on_generic_runtime_error(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    from app.jobs.queue import JobQueue
    from app.jobs.chat_agent import run_chat_agent_job

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = FakeUpstashRedisClient()
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Error session"})
    conversation_id = create.json()["id"]

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "Hello error"},
    )
    assert send.status_code == 202
    job_id = send.json()["job_id"]

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: GenericErrorLLMProvider())
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    job = fake_queue.dequeue("chat_agent_run")
    assert job is not None

    with pytest.raises(ValueError) as excinfo:
        run_chat_agent_job(job.payload)
    
    assert "An unexpected server error occurred." in str(excinfo.value)


def test_send_message_sync_fallback_returns_assistant_message_when_queue_unavailable(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: None)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Sync fallback"})
    conversation_id = create.json()["id"]

    def fake_run_chat_agent_job(payload: dict) -> dict:
        assert payload["job_id"].startswith("sync_")
        return {
            "id": "assistant-sync-1",
            "role": "assistant",
            "content": "Immediate fallback reply",
            "tool_name": None,
            "tool_output": None,
            "created_at": "2026-09-01T00:00:00+00:00",
            "agent_name": "planner",
            "tool_arguments": None,
            "thought_process": "sync fallback completed",
        }

    monkeypatch.setattr("app.jobs.chat_agent.run_chat_agent_job", fake_run_chat_agent_job)

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "Hello without Redis"},
    )

    assert send.status_code == 202
    data = send.json()
    assert data["status"] == "succeeded"
    assert data["job_id"].startswith("sync_")
    assert data["assistant_message"]["content"] == "Immediate fallback reply"
    assert data["assistant_message"]["agent_name"] == "planner"
    assert data["assistant_message"]["thought_process"] == "sync fallback completed"


def test_send_message_sync_fallback_failure_returns_502_when_queue_unavailable(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: None)

    create = client.post("/conversations", headers=auth_headers, json={"title": "Sync fallback fail"})
    conversation_id = create.json()["id"]

    def fake_run_chat_agent_job(payload: dict) -> dict:
        raise ValueError("LLM provider failed: NVIDIA NIM API key is not configured.")

    monkeypatch.setattr("app.jobs.chat_agent.run_chat_agent_job", fake_run_chat_agent_job)

    send = client.post(
        f"/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "Hello without Redis"},
    )

    assert send.status_code == 502
    assert "Failed to process message" in send.json()["detail"]
    assert "NVIDIA NIM API key is not configured" in send.json()["detail"]


def test_send_message_concurrency_lock_prevents_duplicate_sends(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.jobs.queue import JobQueue
    from app.cache.redis_client import RedisCache

    fake_redis = FakeUpstashRedisClient()

    fake_queue = JobQueue("http://fake", "fake")
    fake_queue.client = fake_redis
    monkeypatch.setattr("app.api.routes.conversations.build_job_queue", lambda: fake_queue)

    fake_cache = RedisCache("http://fake", "fake")
    fake_cache.client = fake_redis
    
    from app.api.deps_providers import get_redis_cache
    app.dependency_overrides[get_redis_cache] = lambda: fake_cache
    try:
        create = client.post("/conversations", headers=auth_headers, json={"title": "Concurrency session"})
        conversation_id = create.json()["id"]

        # Send first message - should succeed and be enqueued
        send1 = client.post(
            f"/conversations/{conversation_id}/messages",
            headers=auth_headers,
            json={"content": "First message"},
        )
        assert send1.status_code == 202

        # Send second message immediately - should fail with 409 Conflict because first job is still active (queued)
        send2 = client.post(
            f"/conversations/{conversation_id}/messages",
            headers=auth_headers,
            json={"content": "Second message"},
        )
        assert send2.status_code == 409
        assert "Archimedes is currently answering" in send2.json()["detail"]
        assert len(fake_redis.queues["jobqueue:chat_agent_run"]) == 1
    finally:
        app.dependency_overrides.pop(get_redis_cache, None)

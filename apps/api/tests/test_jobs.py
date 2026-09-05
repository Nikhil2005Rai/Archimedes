import pytest
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.jobs.queue import JobQueue, JobQueueError
from app.jobs.entities import JobStatus
from app.jobs.document_ingestion import run_document_ingestion_job
from app.infrastructure.models import DocumentModel, DocumentChunkModel


class FakeUpstashRedisClient:
    def __init__(self) -> None:
        self.db: dict[str, str] = {}
        self.queues: dict[str, list[str]] = {}

    def get(self, key: str) -> str | None:
        return self.db.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.db[key] = value

    def rpush(self, key: str, value: str) -> int:
        if key not in self.queues:
            self.queues[key] = []
        self.queues[key].append(value)
        return len(self.queues[key])

    def lpop(self, key: str) -> str | None:
        if key not in self.queues or not self.queues[key]:
            return None
        return self.queues[key].pop(0)


class FailingUpstashRedisClient:
    def get(self, key: str) -> str | None:
        raise RuntimeError("Redis connection lost")

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise RuntimeError("Redis connection lost")

    def rpush(self, key: str, value: str) -> int:
        raise RuntimeError("Redis connection lost")

    def lpop(self, key: str) -> str | None:
        raise RuntimeError("Redis connection lost")


class FakeEmbeddingProvider:
    def __init__(self, **kwargs) -> None:
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


def test_job_queue_round_trip() -> None:
    queue = JobQueue("http://fake", "fake")
    queue.client = FakeUpstashRedisClient()

    payload = {"test": 123}
    job = queue.enqueue("test_job", payload)
    assert job.status == JobStatus.QUEUED
    assert job.payload == payload

    retrieved = queue.get(job.id)
    assert retrieved is not None
    assert retrieved.id == job.id

    dequeued = queue.dequeue("test_job")
    assert dequeued is not None
    assert dequeued.id == job.id

    assert queue.dequeue("test_job") is None

    queue.update_status(job.id, JobStatus.RUNNING)
    updated = queue.get(job.id)
    assert updated.status == JobStatus.RUNNING

    queue.update_status(job.id, JobStatus.SUCCEEDED, result={"ok": True})
    updated2 = queue.get(job.id)
    assert updated2.status == JobStatus.SUCCEEDED
    assert updated2.result == {"ok": True}


def test_job_queue_raises_on_failure() -> None:
    queue = JobQueue("http://fake", "fake")
    queue.client = FailingUpstashRedisClient()

    with pytest.raises(JobQueueError):
        queue.enqueue("test_job", {})

    with pytest.raises(JobQueueError):
        queue.dequeue("test_job")

    with pytest.raises(JobQueueError):
        queue.get("some-id")

    with pytest.raises(JobQueueError):
        queue.update_status("some-id", JobStatus.RUNNING)


def test_run_document_ingestion_job(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.jobs.document_ingestion.GeminiEmbeddingProvider",
        FakeEmbeddingProvider
    )
    monkeypatch.setattr(
        "app.jobs.document_ingestion.SessionLocal",
        lambda: db_session
    )
    monkeypatch.setattr(
        "app.jobs.document_ingestion.get_engine",
        lambda: db_session.bind
    )

    from app.infrastructure.models import UserModel
    from app.auth.api_key_repository import UserApiKeyRepository
    from app.auth.encryption import EncryptionService

    user = UserModel(id="test-user-id", name="Test", email="test@example.com", emailVerified=True, createdAt=datetime.now(), updatedAt=datetime.now())
    db_session.add(user)
    db_session.commit()

    UserApiKeyRepository(db_session).upsert(
        user_id="test-user-id",
        provider="gemini",
        encrypted_key=EncryptionService().encrypt("fake-api-key")
    )

    payload = {
        "user_id": "test-user-id",
        "title": "Ingested Notes",
        "content": "alpha beta gamma",
    }

    result = run_document_ingestion_job(payload)
    assert result["title"] == "Ingested Notes"
    assert result["chunk_count"] == 1
    assert "document_id" in result

    document_id = result["document_id"]
    document = db_session.scalar(select(DocumentModel).where(DocumentModel.id == document_id))
    assert document is not None
    assert document.title == "Ingested Notes"
    assert document.user_id == "test-user-id"

    chunk = db_session.scalar(select(DocumentChunkModel).where(DocumentChunkModel.document_id == document.id))
    assert chunk is not None
    assert chunk.content == "alpha beta gamma"
    assert chunk.embedding == [0.1] * 768


def test_run_document_ingestion_job_corrupted_key(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.jobs.document_ingestion.GeminiEmbeddingProvider",
        FakeEmbeddingProvider
    )
    monkeypatch.setattr(
        "app.jobs.document_ingestion.SessionLocal",
        lambda: db_session
    )
    monkeypatch.setattr(
        "app.jobs.document_ingestion.get_engine",
        lambda: db_session.bind
    )

    from app.infrastructure.models import UserModel
    from app.auth.api_key_repository import UserApiKeyRepository

    user = UserModel(id="test-user-id-corrupt", name="Test", email="test-corrupt@example.com", emailVerified=True, createdAt=datetime.now(), updatedAt=datetime.now())
    db_session.add(user)
    db_session.commit()

    # Create key record with corrupted payload that fails decryption
    UserApiKeyRepository(db_session).upsert(
        user_id="test-user-id-corrupt",
        provider="gemini",
        encrypted_key="this-is-corrupted-and-not-valid-fernet"
    )

    payload = {
        "user_id": "test-user-id-corrupt",
        "title": "Ingested Notes",
        "content": "alpha beta gamma",
    }

    with pytest.raises(ValueError, match="Stored API key could not be decrypted"):
        run_document_ingestion_job(payload)


class ScriptedGraphProvider:
    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.calls = []

    def generate(self, messages, tools = None):
        self.calls.append((messages, tools))
        return self.responses.pop(0)


def test_run_chat_agent_job_direct_answer(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.jobs.chat_agent import run_chat_agent_job
    from app.infrastructure.models import UserModel
    from app.auth.api_key_repository import UserApiKeyRepository
    from app.auth.encryption import EncryptionService
    from app.conversations.repository import ConversationRepository
    from app.providers.base import LLMResponse

    user = UserModel(id="chat-user-1", name="Test", email="chat@example.com", emailVerified=True, createdAt=datetime.now(), updatedAt=datetime.now())
    db_session.add(user)
    db_session.commit()

    UserApiKeyRepository(db_session).upsert(
        user_id="chat-user-1",
        provider="gemini",
        encrypted_key=EncryptionService().encrypt("fake-key")
    )

    repo = ConversationRepository(db_session)
    conv = repo.create(user_id="chat-user-1", title="Direct Chat")
    user_msg = repo.add_message(conv.id, "user", "Say hello directly")

    scripted_provider = ScriptedGraphProvider([LLMResponse(content="ANSWER: Direct graph response")])

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: scripted_provider)
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    payload = {
        "conversation_id": conv.id,
        "user_id": "chat-user-1",
        "user_message_id": user_msg.id,
        "content": "Say hello directly"
    }

    result = run_chat_agent_job(payload)
    assert result["role"] == "assistant"
    assert result["content"] == "Direct graph response"

    messages = repo.list_messages(conv.id)
    assert len(messages) == 2
    assert messages[1].role == "assistant"
    assert messages[1].content == "Direct graph response"


def test_run_chat_agent_job_with_images_forces_gemini_provider(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.jobs.chat_agent import run_chat_agent_job
    from app.infrastructure.models import UserModel
    from app.auth.api_key_repository import UserApiKeyRepository
    from app.auth.encryption import EncryptionService
    from app.conversations.repository import ConversationRepository
    from app.providers.base import LLMResponse

    user = UserModel(
        id="chat-user-vision",
        name="Test",
        email="vision@example.com",
        emailVerified=True,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        preferred_provider="nvidia",
        preferred_model="meta/llama-3.1-70b-instruct",
    )
    db_session.add(user)
    db_session.commit()

    UserApiKeyRepository(db_session).upsert(
        user_id="chat-user-vision",
        provider="gemini",
        encrypted_key=EncryptionService().encrypt("fake-gemini-key"),
    )
    UserApiKeyRepository(db_session).upsert(
        user_id="chat-user-vision",
        provider="nvidia",
        encrypted_key=EncryptionService().encrypt("fake-nvidia-key"),
    )

    repo = ConversationRepository(db_session)
    conv = repo.create(user_id="chat-user-vision", title="Vision Chat")
    user_msg = repo.add_message(conv.id, "user", "What is in this image?")

    captured_build_kwargs = {}
    captured_messages = []

    class VisionProvider:
        def generate(self, messages, tools=None):
            captured_messages.extend(messages)
            return LLMResponse(content="This image shows a test fixture.")

    def fake_build_provider(**kwargs):
        captured_build_kwargs.update(kwargs)
        return VisionProvider()

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", fake_build_provider)
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)
    monkeypatch.setattr("app.jobs.chat_agent.build_redis_cache", lambda: None)
    monkeypatch.setattr("app.jobs.chat_agent.build_job_queue", lambda: None)

    payload = {
        "conversation_id": conv.id,
        "user_id": "chat-user-vision",
        "user_message_id": user_msg.id,
        "content": "What is in this image?",
        "images": [{"mime_type": "image/png", "data": "aGVsbG8="}],
    }

    result = run_chat_agent_job(payload)

    assert captured_build_kwargs["provider_name"] == "gemini"
    assert captured_build_kwargs["api_key"] == "fake-gemini-key"
    assert captured_build_kwargs["model"] is None
    assert captured_messages[-1].images is not None
    assert captured_messages[-1].images[0].mime_type == "image/png"
    assert captured_messages[-1].images[0].data == "aGVsbG8="
    assert result["content"] == "This image shows a test fixture."
    assert result["agent_name"] == "gemini"


def test_vision_answer_format_normalizes_latex_units() -> None:
    from app.jobs.chat_agent import _normalize_vision_answer_format

    answer = (
        "Based on the report, the **Reactor Temperature** is **$348\\ ^\\circ\\text{C}$** "
        "(normal range of $340\\text{–}355\\ ^\\circ\\text{C}$)."
    )

    assert _normalize_vision_answer_format(answer) == (
        "Based on the report, the **Reactor Temperature** is **348 \u00b0C** "
        "(normal range of 340–355 \u00b0C)."
    )


def test_run_chat_agent_job_closes_read_session_before_agent_run(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.jobs.chat_agent import run_chat_agent_job
    from app.infrastructure.models import UserModel
    from app.auth.api_key_repository import UserApiKeyRepository
    from app.auth.encryption import EncryptionService
    from app.conversations.repository import ConversationRepository
    from app.agents.planner import PlannerResult

    user = UserModel(id="chat-user-session", name="Test", email="session@example.com", emailVerified=True, createdAt=datetime.now(), updatedAt=datetime.now())
    db_session.add(user)
    db_session.commit()

    UserApiKeyRepository(db_session).upsert(
        user_id="chat-user-session",
        provider="gemini",
        encrypted_key=EncryptionService().encrypt("fake-key")
    )

    repo = ConversationRepository(db_session)
    conv = repo.create(user_id="chat-user-session", title="Session Lifecycle")
    user_msg = repo.add_message(conv.id, "user", "Check session lifecycle")

    class TrackingSession:
        def __init__(self, inner: Session, index: int) -> None:
            self.inner = inner
            self.index = index
            self.closed = False

        def close(self) -> None:
            self.closed = True
            self.inner.close()

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

    created_sessions: list[TrackingSession] = []

    def session_factory() -> TrackingSession:
        session = TrackingSession(db_session, len(created_sessions))
        created_sessions.append(session)
        return session

    class SlowGraph:
        def __init__(self, **kwargs) -> None:
            pass

        def run(self, user_input, history=None):
            assert created_sessions[0].closed is True
            assert created_sessions[1].closed is False
            return PlannerResult(
                answer="Session-safe response",
                tool_name=None,
                agent_name="planner",
                thought_process="phase check",
            )

    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", session_factory)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)
    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: object())
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.MultiAgentGraph", SlowGraph)
    monkeypatch.setattr("app.jobs.chat_agent.build_redis_cache", lambda: None)
    monkeypatch.setattr("app.jobs.chat_agent.build_job_queue", lambda: None)

    payload = {
        "conversation_id": conv.id,
        "user_id": "chat-user-session",
        "user_message_id": user_msg.id,
        "content": "Check session lifecycle",
    }

    result = run_chat_agent_job(payload)

    assert result["content"] == "Session-safe response"
    assert result["agent_name"] == "planner"
    assert result["thought_process"] == "phase check"
    assert [session.closed for session in created_sessions] == [True, True, True]
    messages = repo.list_messages(conv.id)
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Session-safe response"


def test_run_chat_agent_job_routes_to_knowledge(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.jobs.chat_agent import run_chat_agent_job
    from app.infrastructure.models import UserModel
    from app.auth.api_key_repository import UserApiKeyRepository
    from app.auth.encryption import EncryptionService
    from app.conversations.repository import ConversationRepository
    from app.providers.base import LLMResponse
    from app.domain.entities import RetrievedChunk

    user = UserModel(id="chat-user-2", name="Test", email="chat2@example.com", emailVerified=True, createdAt=datetime.now(), updatedAt=datetime.now())
    db_session.add(user)
    db_session.commit()

    UserApiKeyRepository(db_session).upsert(
        user_id="chat-user-2",
        provider="gemini",
        encrypted_key=EncryptionService().encrypt("fake-key")
    )

    repo = ConversationRepository(db_session)
    conv = repo.create(user_id="chat-user-2", title="Knowledge Chat")
    user_msg = repo.add_message(conv.id, "user", "Search my knowledge")

    scripted_provider = ScriptedGraphProvider([
        LLMResponse(content="ROUTE: knowledge"),
        LLMResponse(content="Grounded answer from chunk")
    ])

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: scripted_provider)
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    class ScriptedRetrievalRepository:
        def __init__(self, session):
            pass
        def search(
            self,
            user_id: str,
            embedding: list[float],
            limit: int = 4,
            workspace_id: str | None = None,
        ) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(id="test-chunk-1", document_id="doc-1", content="grounded details", score=0.99)
            ]
        def create(self, conversation_id: str, message_id: str, agent_name: str, query: str, chunk_ids: list[str], scores: list[float]):
            pass

    monkeypatch.setattr("app.jobs.chat_agent.RetrievalRepository", ScriptedRetrievalRepository)

    payload = {
        "conversation_id": conv.id,
        "user_id": "chat-user-2",
        "user_message_id": user_msg.id,
        "content": "Search my knowledge"
    }

    result = run_chat_agent_job(payload)
    assert result["role"] == "assistant"
    assert result["content"] == "Grounded answer from chunk"


def test_run_chat_agent_job_generation_failure(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.jobs.chat_agent import run_chat_agent_job
    from app.infrastructure.models import UserModel
    from app.auth.api_key_repository import UserApiKeyRepository
    from app.auth.encryption import EncryptionService
    from app.conversations.repository import ConversationRepository
    from app.providers.base import LLMGenerationError

    user = UserModel(id="chat-user-3", name="Test", email="chat3@example.com", emailVerified=True, createdAt=datetime.now(), updatedAt=datetime.now())
    db_session.add(user)
    db_session.commit()

    UserApiKeyRepository(db_session).upsert(
        user_id="chat-user-3",
        provider="gemini",
        encrypted_key=EncryptionService().encrypt("fake-key")
    )

    repo = ConversationRepository(db_session)
    conv = repo.create(user_id="chat-user-3", title="Failure Chat")
    user_msg = repo.add_message(conv.id, "user", "This should fail")

    class FailingProvider:
        def __init__(self, **kw):
            pass
        def generate(self, *args, **kwargs):
            raise LLMGenerationError("Upstream LLM timeout")

    monkeypatch.setattr("app.jobs.chat_agent.build_provider", lambda **kw: FailingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.GeminiEmbeddingProvider", lambda **kw: FakeEmbeddingProvider())
    monkeypatch.setattr("app.jobs.chat_agent.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.jobs.chat_agent.get_engine", lambda: db_session.bind)

    payload = {
        "conversation_id": conv.id,
        "user_id": "chat-user-3",
        "user_message_id": user_msg.id,
        "content": "This should fail"
    }

    with pytest.raises(ValueError, match="LLM provider failed: Upstream LLM timeout"):
        run_chat_agent_job(payload)

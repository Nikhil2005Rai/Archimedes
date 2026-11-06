# AI OS Monolith

This is a local-first monolith with two apps:

- `apps/api`: FastAPI backend
- `apps/web`: Next.js frontend

The current foundation rules are intentionally strict:

- One backend process run locally with `uvicorn`
- One frontend process run locally with `next dev`
- Postgres only for persistence
- One active LLM provider behind an abstraction layer
- LangGraph orchestrates Planner, Research, and Knowledge agents
- JWT auth only
- Upstash Redis REST caching for LLM responses and conversation histories
- Redis-backed job queue for asynchronous document ingestion
- No external vector DB, Docker Compose used for AWS EC2 backend deployment (local dev runs bare processes), no complex observability stack

The backend is structured for clean architecture so later agents, retrieval, caching, and external services can be added without a rewrite.

## Agent Graph

The backend uses LangGraph for the multi-agent flow. When a user sends a chat message, the agent run is enqueued as a job, and the graph execution runs asynchronously inside the background worker process:

```text
user request -> planner node -> direct answer
                         |
                         -> research agent -> final answer
                         |
                         -> knowledge agent -> pgvector retrieval -> final answer
```

The Planner decides whether a request should be answered directly, routed to Research, or routed to Knowledge. Research can use the shared `ToolRegistry`. Knowledge embeds the query, searches the current user's document chunks in Postgres/pgvector, and answers from retrieved context.

New agents register in `apps/api/app/agents/registry.py`; graph construction reads from that registry.

## Runbook

Create `apps/api/.env`:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
JWT_SECRET_KEY=replace-with-a-long-random-value
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash
LLM_API_KEY=your-gemini-api-key
OLLAMA_BASE_URL=http://localhost:11434
LLM_MAX_OUTPUT_TOKENS=2048
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
LLM_CACHE_TTL_SECONDS=3600
CONVERSATION_CACHE_TTL_SECONDS=300
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768
ENCRYPTION_KEYS=your-primary-fernet-key
FRONTEND_ORIGIN=http://localhost:3000
```

Generate an encryption key with:

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

`ENCRYPTION_KEYS` is a comma-separated list of Fernet keys, newest (primary) first. New encryptions always use the first key; decryption tries every key in order, so old ciphertexts keep working after rotation.

**Key rotation procedure:**

1. Generate a new key with the command above.
2. Prepend it to `ENCRYPTION_KEYS`, comma-separated: `ENCRYPTION_KEYS=new-key,old-key`
3. Run the rotation script to re-encrypt all stored API keys: `python -m scripts.rotate_encryption_key`
4. Only after the script completes, you may remove the old key(s) from `ENCRYPTION_KEYS`.

> [!WARNING]
> Removing an old key from `ENCRYPTION_KEYS` **before** running the rotation script will make any API key still encrypted with it permanently undecryptable.

> [!NOTE]
> The legacy `ENCRYPTION_KEY` (singular) setting is still supported as a fallback for anyone who hasn't migrated yet. If `ENCRYPTION_KEYS` is empty, the service falls back to `ENCRYPTION_KEY`.

To switch to Groq:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
LLM_API_KEY=your-groq-api-key
```

To switch to local Ollama:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

Ollama must be reachable from the backend process. `localhost` works only when the API backend and Ollama are running on the same machine.

Create `apps/web/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Optional Tracing (LangSmith)

AI-OS supports distributed tracing via [LangSmith](https://smith.langchain.com) to provide deep visibility into LLM calls, graph execution paths, and tool usage. This is entirely optional.

To enable tracing:
1. Create a free account at [smith.langchain.com](https://smith.langchain.com).
2. Generate an API key.
3. Add the following to your `apps/api/.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=ai-os
```

When enabled, traces will include both LangGraph-level routing decisions and the raw HTTP calls made to Gemini/Groq. If disabled or omitted, the application functions normally with zero side effects.

## Local Development:

Run local development with three terminals: backend API, Inngest dev server, and frontend.

### 1. Backend API

```powershell
cd apps/api
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload
```

The API also runs `alembic upgrade head` on startup unless `ENVIRONMENT=test`.

The backend listens on `http://localhost:8000` and serves Inngest functions at:

```text
http://localhost:8000/api/inngest
```

### 2. Inngest Dev Server

Start this after the backend is running:

```powershell
npx --ignore-scripts=false inngest-cli@latest dev --no-discovery -u http://localhost:8000/api/inngest
```

Open the Inngest dashboard at:

```text
http://localhost:8288
```

You should see the `ai-os-api` app and these functions:

```text
chat-agent-run
document-ingestion
```

When chat or document requests are sent from the UI, the backend creates a Redis job for the UI to poll, dispatches the event to Inngest, and the Inngest function updates that same job to `running`, `succeeded`, or `failed`.

### 3. Frontend

```powershell
cd apps/web
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## Deploy on Render Free

The repository includes `render.yaml` for the FastAPI API service. Keep the
frontend deployed on Vercel.

Before deploying, create or connect:

- A PostgreSQL database with the `vector` extension, such as Neon.
- An Upstash Redis database for job state and queue fallback.
- An Inngest Cloud app for production background execution.

After the services are created, set these values in Render:

1. API `DATABASE_URL`, `ENCRYPTION_KEYS`, `LLM_API_KEY`, Redis credentials, and
    Inngest `INNGEST_EVENT_KEY`/`INNGEST_SIGNING_KEY`.
2. In Vercel, set `NEXT_PUBLIC_API_URL` to the Render API service URL, for
    example `https://ai-os-api.onrender.com`, and redeploy the frontend.
3. API `FRONTEND_ORIGIN` to the Vercel production URL.
4. In the Inngest Cloud dashboard, register the sync URL
    `https://ai-os-api.onrender.com/api/inngest`.

Render free web services sleep when idle, so the first request after inactivity
may take several seconds. Do not deploy the local Inngest dev server or the
Redis polling worker as Render free services; Inngest Cloud handles the
background functions in production.

### Optional Worker Fallback

If you are not running the Inngest dev server, run the Redis queue worker instead:

```powershell
cd apps/api
.\.venv\Scripts\Activate.ps1
python -m app.worker
```

> [!IMPORTANT]
> `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` are now **required** (not optional) for all chat and document-upload/ingestion features, as all agent runs and document ingestion operations rely on the Redis-backed job queue. LLM caching remains optional.

Open `http://localhost:3000`, register a user, create a conversation, and send a planner request. The graph can answer directly or route to Research. Tool calls are recorded in `tool_calls`, including the agent that invoked the tool.

For lightweight RAG, paste text into the Knowledge panel. The backend stores `documents` and `document_chunks`, using the Neon Postgres `vector` extension with Gemini embeddings. Ask a knowledge-flavored question in chat and the Planner can route it to the Knowledge agent. Retrieval audits are stored in `retrievals` with retrieved chunk IDs and scores.

BYOK endpoints:

```text
POST /users/me/api-keys
DELETE /users/me/api-keys/{provider}
```

The API never returns stored key material. It only returns provider metadata when a key is set or updated. Saving a key also makes that provider the user's active provider. If a user has exactly one saved key and no explicit preference, that provider is used; otherwise the server default is used.

RAG endpoints:

```text
POST /documents
```

Run backend tests:

```powershell
cd apps/api
.\.venv\Scripts\Activate.ps1
pytest
```

import base64
import binascii
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    emailVerified: bool
    createdAt: datetime
    updatedAt: datetime
    image: str | None = None
    preferred_provider: str | None = None
    preferred_model: str | None = None


class UserPreferencesUpdateRequest(BaseModel):
    preferred_provider: str | None = Field(default=None, max_length=80)
    preferred_model: str | None = Field(default=None, max_length=120)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = Field(default="", max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=160)


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageImageInput(BaseModel):
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    data: str = Field(min_length=1, max_length=7_000_000)

    @field_validator("data")
    @classmethod
    def validate_base64_image_data(cls, value: str) -> str:
        if "," in value and value.strip().startswith("data:"):
            value = value.split(",", 1)[1]
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image data must be valid base64") from exc
        return value


class MessageCreateRequest(BaseModel):
    content: str = Field(default="", max_length=8000)
    images: list[MessageImageInput] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_message_body(self) -> "MessageCreateRequest":
        if not self.content.strip() and not self.images:
            raise ValueError("content or at least one image is required")
        return self


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    tool_name: str | None
    created_at: datetime
    tool_output: str | None = None
    agent_name: str | None = None
    tool_arguments: dict | None = None
    thought_process: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    user_email: str | None = None


class AgentMessageResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


class ApiKeyUpsertRequest(BaseModel):
    provider: Literal["gemini", "groq", "nvidia", "ollama"]
    api_key: str | None = Field(default=None, min_length=1, max_length=4096)
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)

    @model_validator(mode="after")
    def validate_provider_config(self) -> "ApiKeyUpsertRequest":
        if self.provider == "ollama":
            if not self.base_url or not self.base_url.strip():
                raise ValueError("base_url is required for Ollama.")
            return self
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key is required for this provider.")
        return self


class ApiKeyMetadataResponse(BaseModel):
    provider: str
    created_at: datetime


class ApiKeyListResponse(BaseModel):
    providers: list[ApiKeyMetadataResponse]


class DocumentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=200_000)
    chunk_size: int | None = Field(default=None, ge=100, le=5000)
    overlap: int | None = Field(default=None, ge=0, le=1000)


class DocumentResponse(BaseModel):
    id: str
    title: str
    source_type: str
    chunk_count: int
    created_at: datetime


class DocumentJobResponse(BaseModel):
    job_id: str
    status: str


class DocumentJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None


class AgentJobResponse(BaseModel):
    job_id: str
    status: str
    user_message: MessageResponse
    assistant_message: MessageResponse | None = None


class AgentJobStatusResponse(BaseModel):
    job_id: str
    status: str
    assistant_message: MessageResponse | None = None
    execution_steps: list[dict] | None = None
    error: str | None = None


class ShareCreateResponse(BaseModel):
    share_id: str
    title: str


class ShareSnapshotResponse(BaseModel):
    title: str
    messages: list[MessageResponse]


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    owner_id: str
    created_at: datetime
    my_role: str | None = None


class WorkspaceMemberResponse(BaseModel):
    user_id: str
    email: str
    name: str
    role: str
    joined_at: datetime


class WorkspaceMemberUpdateRequest(BaseModel):
    role: str = Field(pattern="^(owner|member|viewer)$")


class WorkspaceInviteRequest(BaseModel):
    email: EmailStr
    role: str = Field(pattern="^(member|viewer)$")


class WorkspaceInviteResponse(BaseModel):
    id: str
    workspace_id: str
    email: str
    role: str
    token: str
    status: str
    expires_at: datetime
    created_at: datetime


class WorkspaceInviteDetailsResponse(BaseModel):
    token: str
    workspace_id: str
    workspace_name: str
    invited_email: str
    role: str
    status: str
    already_member: bool = False
    user_role: str | None = None
    is_owner: bool = False

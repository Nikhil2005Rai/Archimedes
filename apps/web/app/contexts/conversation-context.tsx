"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useRef,
  FormEvent,
  useMemo,
} from "react";
import { useRouter } from "next/navigation";
import { createApiClient } from "../lib/api-client";
import { useAuth } from "./auth-context";
import { useWorkspace } from "./workspace-context";
import { useUi } from "./ui-context";
import { useApiKeys } from "./api-keys-context";
import { pollJob } from "../hooks/use-job-poller";
import toast from "react-hot-toast";

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type ExecutionStep = {
  step: string;
  label: string;
  status: string;
  timestamp: string;
  metadata?: Record<string, any>;
};

export type Message = {
  id: string;
  role: string;
  content: string;
  images?: ChatImageInput[];
  tool_name: string | null;
  tool_output?: string | null;
  agent_name?: string | null;
  tool_arguments?: Record<string, any> | null;
  thought_process?: string | null;
  created_at: string;
  user_id?: string | null;
  user_name?: string | null;
  user_email?: string | null;
  execution_steps?: ExecutionStep[];
};

export type ChatImageInput = {
  mime_type: string;
  data: string;
  preview_url?: string;
  name?: string;
};

export type SuggestionCard = {
  title: string;
  desc: string;
  prompt: string;
};

export const suggestionCards: SuggestionCard[] = [
  {
    title: "Write design specs",
    desc: "Draft a system integration document",
    prompt:
      "Write a high-level system integration design specification for connecting FastAPI with next.js via REST, including error boundaries.",
  },
  {
    title: "Audit database indexes",
    desc: "Recommend indexing strategies for pgvector",
    prompt:
      "Provide an optimal indexing strategy for pgvector HNSW index configurations on high dimension vector models (e.g. 768 dimensions).",
  },
  {
    title: "Optimize API routes",
    desc: "Analyze dependencies and middleware latency",
    prompt:
      "Show how to structure modular FastAPI dependencies to reuse database connections, leverage NullPool correctly, and reduce connection overhead.",
  },
  {
    title: "Draft release notes",
    desc: "Summarize brutalist design system updates",
    prompt:
      "Draft comprehensive release notes explaining the resizable/collapsible retro-brutalist sidebar features, dynamic routes, and SVG icons.",
  },
];

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type ConversationContextType = {
  conversations: Conversation[];
  setConversations: React.Dispatch<React.SetStateAction<Conversation[]>>;
  activeConversationId: string | null;
  setActiveConversationId: (id: string | null) => void;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  currentExecutionSteps: ExecutionStep[];
  editingConversationId: string | null;
  setEditingConversationId: (id: string | null) => void;
  editingTitle: string;
  setEditingTitle: (title: string) => void;
  conversationGroups: [string, Conversation[]][];
  loadConversations: (authToken?: string) => Promise<void>;
  loadMessages: (authToken: string | undefined, conversationId: string) => Promise<void>;
  createConversation: () => Promise<Conversation | null>;
  deleteConversation: (conversationId: string) => Promise<void>;
  renameConversation: (conversationId: string, newTitle: string) => Promise<void>;
  sendMessage: (
    event?: FormEvent<HTMLFormElement>,
    textOverride?: string,
    conversationIdOverride?: string,
    imagesOverride?: ChatImageInput[]
  ) => Promise<void>;
};

const ConversationContext = createContext<ConversationContextType | undefined>(undefined);

export const ConversationProvider = ({ children }: { children: React.ReactNode }) => {
  const router = useRouter();
  const { getToken, isAuthenticated } = useAuth();
  const { activeWorkspaceId } = useWorkspace();
  const { setStatus, setIsSending, setDraft, draft, setIsApiKeyWarningOpen } = useUi();
  const { configuredProviders } = useApiKeys();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentExecutionSteps, setCurrentExecutionSteps] = useState<ExecutionStep[]>([]);
  const [editingConversationId, setEditingConversationId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const sendingRef = useRef(false);

  useEffect(() => {
    if (isAuthenticated && activeWorkspaceId) {
      void loadConversations();
      // Background polling (12s) disabled for now:
      // const interval = setInterval(() => {
      //   if (typeof document !== "undefined" && document.hidden) return;
      //   void loadConversations();
      // }, 12000);
      // return () => clearInterval(interval);
    }
  }, [isAuthenticated, activeWorkspaceId]);

  const conversationGroups = useMemo(() => {
    const today: Conversation[] = [];
    const yesterday: Conversation[] = [];
    const older: Conversation[] = [];

    const now = new Date();
    const oneDay = 24 * 60 * 60 * 1000;

    conversations.forEach((conv) => {
      const convDate = new Date(conv.created_at);
      const diffDays = Math.floor((now.getTime() - convDate.getTime()) / oneDay);

      if (diffDays === 0) {
        today.push(conv);
      } else if (diffDays === 1) {
        yesterday.push(conv);
      } else {
        older.push(conv);
      }
    });

    const groups: [string, Conversation[]][] = [];
    if (today.length > 0) groups.push(["Today", today]);
    if (yesterday.length > 0) groups.push(["Yesterday", yesterday]);
    if (older.length > 0) groups.push(["Older", older]);

    return groups;
  }, [conversations]);

  const api = useMemo(() => createApiClient(getToken), [getToken]);

  const loadConversations = async (authToken?: string) => {
    if (!activeWorkspaceId) return;
    try {
      const activeToken = authToken || (typeof window !== "undefined" ? localStorage.getItem("aios_token") : null);
      if (!activeToken) return;

      const url = `/conversations?workspace_id=${activeWorkspaceId}`;
      const response = await fetch(`${API_URL}${url}`, {
        headers: { Authorization: `Bearer ${activeToken}` },
      });
      if (!response.ok) return;
      const data: Conversation[] = await response.json();
      setConversations(data);
    } catch (error) {
      // Silently handle transient network errors during background polling
    }
  };

  const loadMessages = async (authToken: string | undefined, conversationId: string) => {
    if (!conversationId) return;
    try {
      const activeToken = authToken || (typeof window !== "undefined" ? localStorage.getItem("aios_token") : null);
      if (!activeToken) return;

      const response = await fetch(
        `${API_URL}/conversations/${conversationId}/messages`,
        {
          headers: { Authorization: `Bearer ${activeToken}` },
        }
      );
      if (!response.ok) return;
      const data: Message[] = await response.json();

      const processedData = data.map((msg) => {
        if (
          msg.role === "assistant" &&
          (msg.tool_name || msg.agent_name || msg.thought_process) &&
          !msg.execution_steps
        ) {
          const timestamp = msg.created_at || new Date().toISOString();
          const steps: ExecutionStep[] = [
            {
              step: "planner",
              label: "Planner analyzed prompt",
              status: "completed",
              timestamp,
            },
          ];

          if (msg.thought_process) {
            steps.push({
              step: "thinking",
              label: "Model Reasoning",
              status: "completed",
              timestamp,
              metadata: { thought: msg.thought_process },
            });
          }

          if (msg.agent_name) {
            steps.push({
              step: "specialist",
              label: `Routed to ${msg.agent_name.charAt(0).toUpperCase() + msg.agent_name.slice(1)} Agent`,
              status: "completed",
              timestamp,
              metadata: { agent_name: msg.agent_name },
            });
          }

          if (msg.tool_name) {
            steps.push({
              step: "tool",
              label: `Executed tool \`${msg.tool_name}\``,
              status: "completed",
              timestamp,
              metadata: {
                tool_name: msg.tool_name,
                tool_output: msg.tool_output,
              },
            });
          }

          steps.push({
            step: "finalize",
            label: "Finalized response",
            status: "completed",
            timestamp,
          });

          return {
            ...msg,
            execution_steps: steps,
          };
        }
        return msg;
      });

      // Deduplicate by id — prevents React duplicate-key warning when poll
      // fires while an optimistically-appended message is still in state.
      setMessages((prev) => {
        const incoming = processedData;
        const seen = new Set<string>();
        const merged: typeof incoming = [];
        for (const m of incoming) {
          if (!seen.has(m.id)) { seen.add(m.id); merged.push(m); }
        }
        // Preserve any pending optimistic user messages not yet saved on server
        for (const m of prev) {
          if (m.id.startsWith("user-") && !seen.has(m.id)) {
            merged.push(m);
            seen.add(m.id);
          }
        }
        return merged;
      });
    } catch (error) {
      // Silently handle transient network errors during background polling
    }
  };

  const createConversation = async (): Promise<Conversation | null> => {
    if (!activeWorkspaceId) {
      toast.error("No active workspace selected");
      return null;
    }
    try {
      const newConv = await api<Conversation>(`/conversations?workspace_id=${activeWorkspaceId}`, {
        method: "POST",
        body: JSON.stringify({ title: "New Session" }),
      });
      setConversations((prev) => [newConv, ...prev]);
      setActiveConversationId(newConv.id);
      setMessages([]);
      router.push(`/chat/${newConv.id}`);
      return newConv;
    } catch (error) {
      console.error("Error creating conversation:", error);
      toast.error("Failed to create new conversation");
      return null;
    }
  };

  const deleteConversation = async (conversationId: string) => {
    try {
      await api(`/conversations/${conversationId}`, { method: "DELETE" });
      setConversations((prev) => prev.filter((c) => c.id !== conversationId));
      if (activeConversationId === conversationId) {
        setActiveConversationId(null);
        setMessages([]);
        router.push("/chat");
      }
      toast.success("Conversation deleted");
    } catch (error) {
      console.error("Error deleting conversation:", error);
      toast.error("Failed to delete conversation");
    }
  };

  const renameConversation = async (conversationId: string, newTitle: string) => {
    try {
      const updated = await api<Conversation>(`/conversations/${conversationId}`, {
        method: "PUT",
        body: JSON.stringify({ title: newTitle }),
      });
      setConversations((prev) =>
        prev.map((c) => (c.id === conversationId ? updated : c))
      );
      toast.success("Renamed conversation");
    } catch (error) {
      console.error("Error renaming conversation:", error);
      toast.error("Failed to rename conversation");
    }
  };

  const sendMessage = async (
    event?: FormEvent<HTMLFormElement>,
    textOverride?: string,
    conversationIdOverride?: string,
    imagesOverride?: ChatImageInput[]
  ) => {
    if (event) event.preventDefault();

    if (configuredProviders.length === 0) {
      setIsApiKeyWarningOpen(true);
      return;
    }

    const content = textOverride ?? draft;
    const trimmedContent = content.trim();
    const images = imagesOverride ?? [];
    if (!trimmedContent && images.length === 0) return;

    if (sendingRef.current) return;
    sendingRef.current = true;

    let targetId = conversationIdOverride ?? activeConversationId;
    if (!targetId) {
      const newConv = await createConversation();
      if (!newConv) {
        sendingRef.current = false;
        return;
      }
      targetId = newConv.id;
    }

    setDraft("");
    setIsSending(true);
    setStatus("Thinking...");
    setCurrentExecutionSteps([
      {
        step: "planner",
        label: "Planner analyzing prompt...",
        status: "in_progress",
        timestamp: new Date().toISOString(),
      },
    ]);

    try {
      const token = await getToken();

      const currentConv = conversations.find((c) => c.id === targetId);
      if (
        !currentConv ||
        currentConv.title.toLowerCase().startsWith("new session") ||
        currentConv.title.toLowerCase().startsWith("new chat") ||
        messages.length === 0
      ) {
        const titleSource = trimmedContent || "Image request";
        const autoTitle = titleSource.slice(0, 30) + (titleSource.length > 30 ? "..." : "");
        void renameConversation(targetId, autoTitle);
      }

      // Add optimistic user message
      const userMsgId = `user-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        {
          id: userMsgId,
          role: "user",
          content: trimmedContent || "[Image attached]",
          images,
          tool_name: null,
          created_at: new Date().toISOString(),
        },
      ]);

      const requestImages = images.map(({ mime_type, data }) => ({ mime_type, data }));

      // Post message and retrieve job details
      const response = await api<{ job_id: string; status: string; user_message: Message; assistant_message?: Message | null }>(
        `/conversations/${targetId}/messages`,
        {
          method: "POST",
          body: JSON.stringify({ content: trimmedContent, images: requestImages }),
        }
      );

      const jobId = response.job_id;

      // Optimistically add user message returned from backend (deduped by ID)
      setMessages((prev) => {
        if (prev.some((m) => m.id === response.user_message.id)) {
          return prev.filter((m) => m.id !== userMsgId);
        }
        return [
          ...prev.filter((m) => m.id !== userMsgId),
          { ...response.user_message, images },
        ];
      });

      if (response.assistant_message) {
        const result = response.assistant_message;
        const timestamp = result.created_at || new Date().toISOString();
        const finalSteps: ExecutionStep[] = [
          {
            step: "planner",
            label: "Planner analyzed prompt",
            status: "completed",
            timestamp,
          },
        ];

        if (result.thought_process) {
          finalSteps.push({
            step: "thinking",
            label: "Model Reasoning",
            status: "completed",
            timestamp,
            metadata: { thought: result.thought_process },
          });
        }

        if (result.agent_name) {
          finalSteps.push({
            step: "specialist",
            label: `Routed to ${result.agent_name.charAt(0).toUpperCase() + result.agent_name.slice(1)} Agent`,
            status: "completed",
            timestamp,
            metadata: { agent_name: result.agent_name },
          });
        }

        if (result.tool_name) {
          finalSteps.push({
            step: "tool",
            label: `Executed tool \`${result.tool_name}\``,
            status: "completed",
            timestamp,
            metadata: {
              tool_name: result.tool_name,
              tool_arguments: result.tool_arguments,
              tool_output: result.tool_output,
            },
          });
        }

        finalSteps.push({
          step: "finalize",
          label: "Finalized response",
          status: "completed",
          timestamp,
        });

        setCurrentExecutionSteps(finalSteps);
        setMessages((current) => {
          if (current.some((m) => m.id === result.id)) return current;
          return [
            ...current,
            {
              ...result,
              execution_steps: finalSteps,
            },
          ];
        });
        sendingRef.current = false;
        setIsSending(false);
        setStatus("Ready");
        void loadConversations();
        return;
      }

      // Start polling the job until completed
      pollJob<{
        id: string;
        role: string;
        content: string;
        tool_name: string | null;
        tool_output?: string | null;
        agent_name?: string | null;
        tool_arguments?: Record<string, any> | null;
        thought_process?: string | null;
        created_at: string;
      }>(
        async () => {
          const job = await api<{
            job_id: string;
            status: string;
            assistant_message: {
              id: string;
              role: string;
              content: string;
              tool_name: string | null;
              tool_output?: string | null;
              agent_name?: string | null;
              tool_arguments?: Record<string, any> | null;
              thought_process?: string | null;
              created_at: string;
            } | null;
            execution_steps: ExecutionStep[] | null;
            error: string | null;
          }>(`/conversations/${targetId}/messages/jobs/${jobId}`);

          const succeeded = job.status === "succeeded";
          const failed = job.status === "failed";

          if (job.status === "queued") {
            setStatus("Queued...");
          } else if (job.status === "running") {
            setStatus("Orchestrating agents...");
          }

          if (job.execution_steps && job.execution_steps.length > 0) {
            setCurrentExecutionSteps(job.execution_steps);
          }

          return {
            status: job.status,
            succeeded,
            failed,
            data: job.assistant_message ?? undefined,
            error: job.error ?? undefined,
          };
        },
        {
          intervalMs: 1500,
          timeoutMs: 600000,
          onTimeout: () => {
            setStatus("Request timed out");
            sendingRef.current = false;
            setIsSending(false);
          },
          onSucceeded: (result) => {
            let finalSteps: ExecutionStep[] = [];
            if (result) {
              const timestamp = result.created_at || new Date().toISOString();

              finalSteps.push({
                step: "planner",
                label: "Planner analyzed prompt",
                status: "completed",
                timestamp,
              });

              if (result.thought_process) {
                finalSteps.push({
                  step: "thinking",
                  label: "Model Reasoning",
                  status: "completed",
                  timestamp,
                  metadata: { thought: result.thought_process },
                });
              }

              if (result.agent_name) {
                finalSteps.push({
                  step: "specialist",
                  label: `Routed to ${result.agent_name.charAt(0).toUpperCase() + result.agent_name.slice(1)} Agent`,
                  status: "completed",
                  timestamp,
                  metadata: { agent_name: result.agent_name },
                });
              }

              if (result.tool_name) {
                finalSteps.push({
                  step: "tool",
                  label: `Executed tool \`${result.tool_name}\``,
                  status: "completed",
                  timestamp,
                  metadata: {
                    tool_name: result.tool_name,
                    tool_arguments: result.tool_arguments,
                    tool_output: result.tool_output,
                  },
                });
              }

              finalSteps.push({
                step: "finalize",
                label: "Finalized response",
                status: "completed",
                timestamp,
              });

              setCurrentExecutionSteps(finalSteps);

              setMessages((current) => {
                // Don't add if already present (e.g. polling already fetched it)
                if (current.some((m) => m.id === result.id)) return current;
                return [
                  ...current,
                  {
                    id: result.id,
                    role: result.role,
                    content: result.content,
                    tool_name: result.tool_name,
                    tool_output: result.tool_output,
                    created_at: result.created_at,
                    execution_steps: finalSteps,
                  },
                ];
              });
            }
            sendingRef.current = false;
            setIsSending(false);
            setStatus("Ready");
            void loadConversations();
          },
          onFailed: (error) => {
            setStatus(`Failed: ${error}`);
            toast.error(`Request failed: ${error}`);
            sendingRef.current = false;
            setIsSending(false);
          },
        }
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Request failed";
      setStatus(message);
      toast.error(`Error: ${message}`);
      sendingRef.current = false;
      setIsSending(false);
    }
  };

  return (
    <ConversationContext.Provider
      value={{
        conversations,
        setConversations,
        activeConversationId,
        setActiveConversationId,
        messages,
        setMessages,
        currentExecutionSteps,
        editingConversationId,
        setEditingConversationId,
        editingTitle,
        setEditingTitle,
        conversationGroups,

        loadConversations,
        loadMessages,
        createConversation,
        deleteConversation,
        renameConversation,
        sendMessage,
      }}
    >
      {children}
    </ConversationContext.Provider>
  );
};

export const useConversation = () => {
  const context = useContext(ConversationContext);
  if (!context) {
    throw new Error("useConversation must be used within a ConversationProvider");
  }
  return context;
};

export const useConversations = useConversation;

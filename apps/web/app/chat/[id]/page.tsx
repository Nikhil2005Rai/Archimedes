"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import { useChat } from "../../chat-context";
import { useAuth } from "../../contexts/auth-context";
import { User, Bot, Wrench, Copy, ThumbsUp, ThumbsDown, PanelRightOpen, Users, ChevronUp, ChevronDown, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ExecutionTrace } from "../../components/execution-trace";
import { MermaidDiagram } from "../../components/mermaid-diagram";
import { SpeechPlayer } from "../../components/voice-input";
import { ArtifactDrawer } from "../../components/artifact-drawer";
import { ExportShareModal } from "../../components/export-share-modal";
import { ChartRenderer } from "../../components/chart-renderer";
import { HighlightText } from "../../components/message-search-highlight";

export default function ChatSessionPage() {
  const params = useParams();
  const chatId = params.id as string;
  const { user } = useAuth();

  const [artifact, setArtifact] = useState<{ title: string; language: string; content: string } | null>(null);
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [searchMatchIndex, setSearchMatchIndex] = useState(0);

  const {
    token,
    messages,
    conversations,
    isSending,
    currentExecutionSteps,
    setActiveConversationId,
    loadMessages,
    searchQuery,
    setSearchQuery,
  } = useChat();

  const currentConv = conversations.find((c) => c.id === chatId);
  const messageRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    if (chatId) {
      setActiveConversationId(chatId);
      void loadMessages(token || undefined, chatId);
    }
  }, [chatId, token]);

  useEffect(() => {
    setSearchMatchIndex(0);
  }, [searchQuery]);

  const uniqueMessages = React.useMemo(() => {
    const seen = new Set<string>();
    return messages.filter((msg) => {
      if (!msg.id || seen.has(msg.id)) return false;
      seen.add(msg.id);
      return true;
    });
  }, [messages]);

  const matchingIds = React.useMemo(() => {
    if (!searchQuery.trim()) return [];
    const q = searchQuery.toLowerCase();
    return uniqueMessages
      .filter((m) => m.content.toLowerCase().includes(q))
      .map((m) => m.id);
  }, [uniqueMessages, searchQuery]);

  useEffect(() => {
    if (matchingIds.length === 0) return;
    const targetId = matchingIds[searchMatchIndex];
    const el = messageRefs.current.get(targetId);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [searchMatchIndex, matchingIds]);

  const goPrev = useCallback(() => {
    setSearchMatchIndex((i) => (i - 1 + matchingIds.length) % matchingIds.length);
  }, [matchingIds.length]);

  const goNext = useCallback(() => {
    setSearchMatchIndex((i) => (i + 1) % matchingIds.length);
  }, [matchingIds.length]);

  const renderMarkdownComponents = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || "");
      const lang = match ? match[1] : "";
      const codeString = String(children).replace(/\n$/, "");

      if (!inline && lang === "mermaid") {
        return <MermaidDiagram chart={codeString} />;
      }

      if (!inline && (lang === "chart" || className?.includes("json:chart") || codeString.includes('"chart_type":'))) {
        return <ChartRenderer jsonContent={codeString} />;
      }

      if (!inline && codeString.length > 80) {
        return (
          <div style={{ position: "relative" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "#1e293b", color: "#94a3b8", padding: "4px 12px", borderRadius: "6px 6px 0 0", fontSize: "11px" }}>
              <span>{lang || "code"}</span>
              <button
                type="button"
                onClick={() => setArtifact({ title: "Code Snippet", language: lang || "code", content: codeString })}
                style={{ background: "none", border: "none", color: "#38bdf8", cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}
              >
                <PanelRightOpen size={12} />
                <span>Open Canvas</span>
              </button>
            </div>
            <pre className={className} {...props} style={{ marginTop: 0, borderRadius: "0 0 6px 6px" }}>
              <code>{children}</code>
            </pre>
          </div>
        );
      }

      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
  };

  return (
    <div className="messages-container" style={{ display: "flex", width: "100%", flexDirection: "column" }}>

      {searchQuery.trim() && (
        <div className="search-nav-bar">
          <span className="search-nav-count">
            {matchingIds.length === 0
              ? "No results"
              : `${searchMatchIndex + 1} of ${matchingIds.length}`}
          </span>
          <button type="button" className="search-nav-btn" onClick={goPrev} disabled={matchingIds.length === 0} title="Previous match">
            <ChevronUp size={13} />
          </button>
          <button type="button" className="search-nav-btn" onClick={goNext} disabled={matchingIds.length === 0} title="Next match">
            <ChevronDown size={13} />
          </button>
          <button type="button" className="search-nav-btn search-nav-close" onClick={() => setSearchQuery("")} title="Clear search">
            <X size={13} />
          </button>
        </div>
      )}

      <div className="messages" style={{ flex: 1 }}>

        {uniqueMessages.map((message) => {
          const isAssistant = message.role === "assistant";
          const isCurrentUser = message.user_id
            ? message.user_id === user?.id
            : message.role === "user";

          const isMatch = searchQuery.trim()
            ? message.content.toLowerCase().includes(searchQuery.toLowerCase())
            : false;

          const isActiveMatch = isMatch && matchingIds[searchMatchIndex] === message.id;

          let senderName = "Archimedes";
          if (!isAssistant) {
            if (isCurrentUser) {
              senderName = "You";
            } else {
              senderName = message.user_name || message.user_email || "Team Member";
            }
          }

          return (
            <div
              key={message.id}
              ref={(el) => {
                if (el) messageRefs.current.set(message.id, el);
                else messageRefs.current.delete(message.id);
              }}
              className={`message-group ${
                isAssistant ? "assistant-group" : isCurrentUser ? "user-group" : "teammate-group user-group"
              }${isMatch ? " search-match-highlighted" : ""}${isActiveMatch ? " search-match-active" : ""}`}
            >
              <div
                className={`message-avatar ${
                  isAssistant ? "assistant-avatar" : isCurrentUser ? "user-avatar" : "teammate-avatar"
                }`}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: !isAssistant && !isCurrentUser ? "#0284c7" : undefined,
                  color: "#ffffff",
                }}
              >
                {isAssistant ? <Bot size={16} /> : isCurrentUser ? <User size={16} /> : <Users size={16} />}
              </div>

              <div className="message-bubble-wrapper">
                {!isAssistant && message.execution_steps && message.execution_steps.length > 0 && (
                  <ExecutionTrace steps={message.execution_steps} isPending={false} />
                )}

                <article className="message-bubble">
                  <span style={{ fontWeight: 700, color: !isAssistant && !isCurrentUser ? "#0284c7" : undefined }}>
                    {senderName}
                  </span>

                  {isAssistant && message.tool_name === "chart_generator" && message.tool_output && !message.content.includes('"chart_type"') && (
                    <ChartRenderer jsonContent={message.tool_output} />
                  )}

                  {!isAssistant && message.images && message.images.length > 0 && (
                    <div className="message-image-grid">
                      {message.images.map((image, index) => (
                        <img
                          key={`${image.name || "image"}-${index}`}
                          src={image.preview_url || `data:${image.mime_type};base64,${image.data}`}
                          alt={image.name || `Attached image ${index + 1}`}
                        />
                      ))}
                    </div>
                  )}

                  <div className="message-markdown">
                    {searchQuery.trim() && isMatch ? (
                      <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                        <HighlightText text={message.content} query={searchQuery} />
                      </div>
                    ) : (
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={renderMarkdownComponents}>
                        {message.content}
                      </ReactMarkdown>
                    )}
                  </div>

                  {message.tool_name && message.tool_name !== "current_time" && (
                    <div className="tool-pill" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span className="tool-icon" style={{ display: "flex", alignItems: "center" }}>
                        <Wrench size={12} />
                      </span>
                      <span>Used Tool: <code>{message.tool_name}</code></span>
                    </div>
                  )}
                </article>

                {isAssistant && (
                  <div className="message-actions">
                    <button
                      type="button"
                      className="action-btn"
                      title="Copy output"
                      onClick={() => navigator.clipboard.writeText(message.content)}
                      style={{ display: "flex", alignItems: "center", gap: 4 }}
                    >
                      <Copy size={12} />
                      <span>Copy</span>
                    </button>

                    <SpeechPlayer text={message.content} />

                    <button type="button" className="action-btn" title="Helpful" style={{ display: "flex", alignItems: "center" }}>
                      <ThumbsUp size={12} />
                    </button>
                    <button type="button" className="action-btn" title="Not helpful" style={{ display: "flex", alignItems: "center" }}>
                      <ThumbsDown size={12} />
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {isSending && (
          <div className="message-group assistant-group pending-group">
            <div className="message-avatar assistant-avatar" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Bot size={16} />
            </div>
            <div className="message-bubble-wrapper" style={{ width: "100%" }}>
              {currentExecutionSteps.length > 0 ? (
                <ExecutionTrace steps={currentExecutionSteps} isPending={true} />
              ) : (
                <article className="message-bubble pending-bubble">
                  <span>Archimedes</span>
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <p className="pending-text">Planner is reasoning...</p>
                </article>
              )}
            </div>
          </div>
        )}
      </div>

      <ArtifactDrawer
        isOpen={!!artifact}
        onClose={() => setArtifact(null)}
        title={artifact?.title || "Artifact"}
        language={artifact?.language || "text"}
        content={artifact?.content || ""}
      />

      <ExportShareModal
        isOpen={isExportOpen}
        onClose={() => setIsExportOpen(false)}
        title={currentConv?.title || "Archimedes Chat"}
        messages={messages}
        conversationId={chatId}
      />
    </div>
  );
}

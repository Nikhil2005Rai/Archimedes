"use client";

import React from "react";
import { AuthProvider, useAuth } from "./contexts/auth-context";
import { WorkspaceProvider, useWorkspace } from "./contexts/workspace-context";
import { ApiKeysProvider, useApiKeys } from "./contexts/api-keys-context";
import { DocumentsProvider, useDocuments } from "./contexts/documents-context";
import { UiProvider, useUi } from "./contexts/ui-context";
import {
  ConversationProvider,
  useConversations,
  suggestionCards,
} from "./contexts/conversation-context";

// Re-export types & constants so existing imports keep working
export type { ChatImageInput, Conversation, Message, SuggestionCard } from "./contexts/conversation-context";
export { suggestionCards } from "./contexts/conversation-context";
export type { DocumentBase } from "./contexts/documents-context";
export type { Workspace, WorkspaceMember } from "./contexts/workspace-context";

export const ChatProvider = ({ children }: { children: React.ReactNode }) => (
  <AuthProvider>
    <WorkspaceProvider>
      <ApiKeysProvider>
        <UiProvider>
          <DocumentsProvider>
            <ConversationProvider>{children}</ConversationProvider>
          </DocumentsProvider>
        </UiProvider>
      </ApiKeysProvider>
    </WorkspaceProvider>
  </AuthProvider>
);

/** Merged hook — every component that calls useChat() keeps working with zero changes. */
export function useChat() {
  const auth = useAuth();
  const workspace = useWorkspace();
  const apiKeys = useApiKeys();
  const documents = useDocuments();
  const ui = useUi();
  const conversations = useConversations();

  return {
    // auth
    ...auth,
    // workspace
    ...workspace,
    // api keys
    ...apiKeys,
    // documents
    ...documents,
    // ui
    ...ui,
    // conversations
    ...conversations,
    // Legacy aliases expected by some callers
    suggestionCards,
  };
}

/**
 * SessionList Component
 *
 * Sidebar displaying chat session history
 */

"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  MessageSquare,
  Plus,
  Trash2,
  Edit2,
  Check,
  X,
  Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ChatSession } from "@/hooks/useSessionManager";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface SessionListProps {
  sessions: ChatSession[];
  currentSession: ChatSession | null;
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (sessionId: string) => void;
  onUpdateTitle: (sessionId: string, title: string) => void;
  loading?: boolean;
}

export function SessionList({
  sessions,
  currentSession,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onUpdateTitle,
  loading = false,
}: SessionListProps) {
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [deleteDialogSessionId, setDeleteDialogSessionId] = useState<string | null>(null);

  const startEditing = (session: ChatSession) => {
    setEditingSessionId(session.session_id);
    setEditingTitle(session.title);
  };

  const cancelEditing = () => {
    setEditingSessionId(null);
    setEditingTitle("");
  };

  const saveTitle = (sessionId: string) => {
    if (editingTitle.trim()) {
      onUpdateTitle(sessionId, editingTitle.trim());
    }
    cancelEditing();
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="w-64 border-r bg-sidebar/50 backdrop-blur-sm h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b">
        <Button
          onClick={onCreateSession}
          className="w-full gap-2"
          disabled={loading}
        >
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
      </div>

      {/* Session List */}
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {loading && sessions.length === 0 ? (
            <div className="text-center text-muted-foreground text-sm py-8">
              Loading sessions...
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center text-muted-foreground text-sm py-8">
              <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No sessions yet</p>
              <p className="text-xs mt-1">Start a new chat to begin</p>
            </div>
          ) : (
            sessions.map((session) => {
              const isActive = currentSession?.session_id === session.session_id;
              const isEditing = editingSessionId === session.session_id;

              return (
                <div
                  key={session.session_id}
                  className={cn(
                    "group relative rounded-md transition-colors",
                    isActive
                      ? "bg-sidebar-accent"
                      : "hover:bg-sidebar-accent/50"
                  )}
                >
                  {isEditing ? (
                    <div className="flex items-center gap-1 p-2">
                      <Input
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            saveTitle(session.session_id);
                          } else if (e.key === "Escape") {
                            cancelEditing();
                          }
                        }}
                        className="h-7 text-sm"
                        autoFocus
                      />
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7 flex-shrink-0"
                        onClick={() => saveTitle(session.session_id)}
                      >
                        <Check className="h-3 w-3" />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7 flex-shrink-0"
                        onClick={cancelEditing}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                  ) : (
                    <button
                      onClick={() => onSelectSession(session.session_id)}
                      className="w-full text-left p-2 pr-16"
                    >
                      <div className="flex items-start gap-2">
                        <MessageSquare className="h-4 w-4 mt-0.5 flex-shrink-0 text-muted-foreground" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">
                            {session.title}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <Clock className="h-3 w-3 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground">
                              {formatDate(session.updated_at)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                            <span>{session.message_count} messages</span>
                            {session.total_tokens > 0 && (
                              <>
                                <span>•</span>
                                <span>{session.total_tokens.toLocaleString()} tokens</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    </button>
                  )}

                  {!isEditing && (
                    <div className="absolute right-2 top-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7"
                        onClick={(e) => {
                          e.stopPropagation();
                          startEditing(session);
                        }}
                      >
                        <Edit2 className="h-3 w-3" />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7 text-destructive hover:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteDialogSessionId(session.session_id);
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>

      {/* Delete Confirmation Dialog */}
      <AlertDialog
        open={deleteDialogSessionId !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteDialogSessionId(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete session?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this chat session and all its messages.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deleteDialogSessionId) {
                  onDeleteSession(deleteDialogSessionId);
                  setDeleteDialogSessionId(null);
                }
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

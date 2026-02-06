"use client";

import { useState } from "react";
import { PageLayout } from "@/components/PageLayout";
import { ApiKeyModal } from "@/components/ApiKeyModal";
import { GoogleConnectModal } from "@/components/GoogleConnectModal";
import { Button } from "@/components/ui/button";
import {
  Key,
  Chrome,
  Info,
  ExternalLink,
  Github,
  Book,
  ArrowRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function SettingsPage() {
  const [apiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  const [googleModalOpen, setGoogleModalOpen] = useState(false);
  const [googleConnected, setGoogleConnected] = useState(false);

  return (
    <PageLayout title="Settings" description="API keys, Google, and system info">
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
          <div className="space-y-4">
            {/* API Keys — Chat-style card: border only, hover */}
            <button
              type="button"
              onClick={() => setApiKeyModalOpen(true)}
              className={cn(
                "w-full group flex items-start gap-3 p-3 rounded-lg border border-border/50 bg-background hover:bg-muted/30 text-left",
                "focus:outline-none focus:ring-2 focus:ring-primary/30"
              )}
            >
              <Key className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" strokeWidth={1.5} />
              <div className="flex-1 min-w-0">
                <span className="text-sm font-medium text-foreground block">API Keys</span>
                <span className="text-xs text-muted-foreground line-clamp-2 block mt-0.5">
                  Manage LLM and external service API keys. Stored encrypted.
                </span>
              </div>
              <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 mt-1" strokeWidth={1.5} />
            </button>

            {/* Google — same card style */}
            <button
              type="button"
              onClick={() => setGoogleModalOpen(true)}
              className={cn(
                "w-full group flex items-start gap-3 p-3 rounded-lg border border-border/50 bg-background hover:bg-muted/30 text-left",
                "focus:outline-none focus:ring-2 focus:ring-primary/30"
              )}
            >
              <Chrome className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" strokeWidth={1.5} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-medium text-foreground block">Google Workspace</span>
                  {googleConnected && (
                    <span className="text-xs font-medium text-primary">Connected</span>
                  )}
                </div>
                <span className="text-xs text-muted-foreground line-clamp-2 block mt-0.5">
                  Connect Drive, Docs, Gmail and more to save and share chats.
                </span>
              </div>
              <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 mt-1" strokeWidth={1.5} />
            </button>

            {/* Info — Chat-style card, no hover action */}
            <div className="rounded-lg border border-border/50 bg-muted/5 overflow-hidden">
              <div className="py-3 px-4 flex items-center gap-2 border-b border-border/30">
                <Info className="h-4 w-4 shrink-0 text-muted-foreground" strokeWidth={1.5} />
                <span className="text-sm font-medium text-foreground tracking-tight">
                  BeanLLM Playground
                </span>
              </div>
              <div className="p-4 space-y-3">
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-muted-foreground block">Version</span>
                    <span className="font-medium text-foreground">1.0.0</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block">Framework</span>
                    <span className="font-medium text-foreground">Next.js 15 + FastAPI</span>
                  </div>
                </div>
                <div className="flex items-center gap-3 pt-3 border-t border-border/30">
                  <a
                    href="https://github.com/leebeanbin/beanllm"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <Github className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
                    GitHub
                    <ExternalLink className="h-3 w-3 shrink-0" />
                  </a>
                  <a
                    href="https://github.com/leebeanbin/beanllm/wiki"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <Book className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
                    Docs
                    <ExternalLink className="h-3 w-3 shrink-0" />
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <ApiKeyModal open={apiKeyModalOpen} onOpenChange={setApiKeyModalOpen} />
      <GoogleConnectModal
        open={googleModalOpen}
        onOpenChange={setGoogleModalOpen}
        userId="default"
        onAuthChange={setGoogleConnected}
      />
    </PageLayout>
  );
}

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
  Wrench,
} from "lucide-react";

export default function SettingsPage() {
  const [apiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  const [googleModalOpen, setGoogleModalOpen] = useState(false);
  const [googleConnected, setGoogleConnected] = useState(false);

  return (
    <PageLayout
      title="Settings"
      description="Manage API keys, Google integration, and system info in one place"
    >
      <div className="space-y-4">
        {/* API Keys */}
        <section className="border border-border/40 rounded-lg overflow-hidden">
          <div className="py-3 px-3 border-b border-border/40 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Key className="h-4 w-4 shrink-0 text-muted-foreground" />
              <span className="text-[13px] font-semibold text-foreground tracking-tight">API Keys</span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setApiKeyModalOpen(true)}
              className="h-8 text-[12px] font-medium gap-1.5"
            >
              <Wrench className="h-3.5 w-3.5 shrink-0" />
              Open
            </Button>
          </div>
          <p className="text-[12px] text-muted-foreground px-3 py-2">
            Manage LLM and external service API keys. Stored encrypted.
          </p>
        </section>

        {/* Google */}
        <section className="border border-border/40 rounded-lg overflow-hidden">
          <div className="py-3 px-3 border-b border-border/40 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Chrome className="h-4 w-4 shrink-0 text-muted-foreground" />
              <span className="text-[13px] font-semibold text-foreground tracking-tight">
                Google Workspace
              </span>
              {googleConnected && (
                <span className="text-[11px] text-green-600 font-medium">Connected</span>
              )}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setGoogleModalOpen(true)}
              className="h-8 text-[12px] font-medium gap-1.5"
            >
              <Wrench className="h-3.5 w-3.5 shrink-0" />
              Open
            </Button>
          </div>
          <p className="text-[12px] text-muted-foreground px-3 py-2">
            Connect Drive, Docs, Gmail and more to save and share chats.
          </p>
        </section>

        {/* Info */}
        <section className="border border-border/40 rounded-lg overflow-hidden">
          <div className="py-3 px-3 border-b border-border/40 flex items-center gap-2">
            <Info className="h-4 w-4 shrink-0 text-muted-foreground" />
            <span className="text-[13px] font-semibold text-foreground tracking-tight">
              BeanLLM Playground
            </span>
          </div>
          <div className="p-3 space-y-3">
            <div className="grid grid-cols-2 gap-2 text-[12px]">
              <div>
                <span className="text-muted-foreground">Version</span>
                <div className="font-medium text-foreground">1.0.0</div>
              </div>
              <div>
                <span className="text-muted-foreground">Framework</span>
                <div className="font-medium text-foreground">Next.js 15 + FastAPI</div>
              </div>
            </div>
            <div className="flex items-center gap-3 pt-2 border-t border-border/40">
              <a
                href="https://github.com/leebeanbin/beanllm"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-[12px] text-muted-foreground hover:text-foreground"
              >
                <Github className="h-3.5 w-3.5 shrink-0" />
                GitHub
                <ExternalLink className="h-3 w-3 shrink-0" />
              </a>
              <a
                href="https://github.com/leebeanbin/beanllm/wiki"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-[12px] text-muted-foreground hover:text-foreground"
              >
                <Book className="h-3.5 w-3.5 shrink-0" />
                Docs
                <ExternalLink className="h-3 w-3 shrink-0" />
              </a>
            </div>
          </div>
        </section>
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

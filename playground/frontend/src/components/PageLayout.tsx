"use client";

import { ReactNode } from "react";
import { Github } from "lucide-react";
import { BrowserTabs } from "./BrowserTabs";

interface PageLayoutProps {
  children: ReactNode;
  title: string;
  description?: string;
  /** Renders at the right end of the top tab bar (e.g. chat page panel toggle) */
  headerTrailing?: ReactNode;
}

const GITHUB_URL = "https://github.com/leebeanbin/beanllm";

export function PageLayout({ children, title, description, headerTrailing }: PageLayoutProps) {
  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      <div className="flex-shrink-0 w-full">
        <BrowserTabs trailing={headerTrailing} />
      </div>
      <main className="flex-1 min-h-0 overflow-hidden" role="main" aria-label={title}>
        {children}
      </main>
      {/* Footer: GitHub only — Settings is in the top tabs */}
      <footer className="flex-shrink-0 flex items-center justify-between gap-4 px-4 py-1.5 border-t border-border/40 bg-background text-xs text-muted-foreground">
        <div className="flex items-center gap-3">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
            aria-label="Open GitHub repository"
          >
            <Github className="h-3.5 w-3.5" strokeWidth={1.5} />
            <span>GitHub</span>
          </a>
        </div>
        <span className="text-[10px] text-muted-foreground/70">beanllm Playground</span>
      </footer>
    </div>
  );
}

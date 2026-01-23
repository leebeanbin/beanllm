"use client";

import { ReactNode } from "react";
import { Navigation } from "./Navigation";

interface PageLayoutProps {
  children: ReactNode;
  title: string;
  description?: string;
}

export function PageLayout({ children, title, description }: PageLayoutProps) {
  return (
    <div className="flex h-screen bg-background">
      <Navigation />
      <main className="flex-1 overflow-auto bg-background/50" role="main" aria-label={title}>
        <div className="container mx-auto p-6 max-w-6xl">
          <header className="mb-6">
            <h1 className="text-3xl font-semibold text-foreground">{title}</h1>
            {description && (
              <p className="text-muted-foreground mt-2 text-sm" role="doc-subtitle">{description}</p>
            )}
          </header>
          {children}
        </div>
      </main>
    </div>
  );
}

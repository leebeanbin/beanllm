"use client";

import { usePathname, useRouter } from "next/navigation";
import { MessageSquare, Activity, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

interface Tab {
  id: string;
  label: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
}

const tabs: Tab[] = [
  { id: "chat", label: "Chat", path: "/chat", icon: MessageSquare },
  { id: "monitoring", label: "Monitoring", path: "/monitoring", icon: Activity },
  { id: "settings", label: "Settings", path: "/settings", icon: Settings },
];

interface BrowserTabsProps {
  trailing?: React.ReactNode;
}

export function BrowserTabs({ trailing }: BrowserTabsProps) {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <div className="flex items-center justify-between gap-2 border-b border-border/40 bg-muted/5 px-3 pt-2 pb-0 flex-shrink-0 antialiased w-full">
      <div className="flex items-center gap-1">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = pathname === tab.path;
        return (
          <button
            key={tab.id}
            onClick={() => router.push(tab.path)}
            className={cn(
              "flex items-center gap-2 px-4 py-2.5 text-[13px] font-medium tracking-tight transition-all rounded-t-lg -mb-px",
              isActive
                ? "bg-background text-foreground border border-border/40 border-b-0"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/10 border border-transparent"
            )}
          >
            <Icon className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
            <span>{tab.label}</span>
          </button>
        );
      })}
      </div>
      {trailing != null ? <div className="flex items-center shrink-0">{trailing}</div> : null}
    </div>
  );
}

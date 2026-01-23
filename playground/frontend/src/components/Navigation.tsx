"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { MessageSquare, Home } from "lucide-react";
import { BeanIcon } from "./icons/BeanIcon";

interface NavItem {
  name: string;
  href: string;
  icon: any;
  description?: string;
}

const navigationItems: NavItem[] = [
  { name: "Home", href: "/", icon: Home, description: "홈" },
  {
    name: "Unified Chat",
    href: "/chat",
    icon: MessageSquare,
    description: "Chat, RAG, Multi-Agent, KG, Audio, OCR, Google, Web Search",
  },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav
      className="w-64 border-r bg-sidebar/50 backdrop-blur-sm h-screen overflow-y-auto"
      aria-label="사이드 네비게이션"
      role="navigation"
    >
      <div className="p-4 pb-6 border-b border-sidebar-border/50">
        <Link
          href="/"
          className="flex items-center gap-2.5 group"
          aria-label="홈으로 이동"
        >
          <div className="w-8 h-8 flex items-center justify-center text-primary transition-transform group-hover:scale-110">
            <BeanIcon className="w-full h-full" aria-hidden="true" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-sidebar-foreground">
              BeanLLM
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">Playground</p>
          </div>
        </Link>
      </div>

      <div className="p-2 space-y-1" role="list">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              role="listitem"
              aria-current={isActive ? "page" : undefined}
            >
              <Button
                variant="ghost"
                className={cn(
                  "w-full justify-start h-auto py-3 px-3 text-sm font-normal transition-all flex-col items-start gap-1",
                  "hover:bg-sidebar-accent/70 hover:text-sidebar-accent-foreground",
                  "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-sm"
                    : "text-sidebar-foreground/70"
                )}
              >
                <div className="flex items-center gap-2.5 w-full">
                  <Icon
                    className={cn(
                      "h-4 w-4 transition-colors flex-shrink-0",
                      isActive ? "text-sidebar-primary" : "text-muted-foreground"
                    )}
                    aria-hidden="true"
                  />
                  <span className="truncate font-medium">{item.name}</span>
                </div>
                {item.description && (
                  <span className="text-xs text-muted-foreground pl-6 text-left">
                    {item.description}
                  </span>
                )}
              </Button>
            </Link>
          );
        })}
      </div>

      <div className="p-4 border-t border-sidebar-border/50 mt-auto">
        <div className="text-xs text-muted-foreground space-y-1">
          <p className="font-medium">Features (Unified Chat):</p>
          <ul className="list-disc list-inside space-y-0.5 ml-1">
            <li>Chat - 일반 대화</li>
            <li>RAG - 문서 검색</li>
            <li>Multi-Agent - 협업</li>
            <li>Knowledge Graph - 지식 그래프</li>
            <li>Audio - 음성 인식</li>
            <li>OCR - 텍스트 인식</li>
            <li>Google - Workspace 연동</li>
            <li>Web Search - 웹 검색</li>
          </ul>
        </div>
      </div>
    </nav>
  );
}

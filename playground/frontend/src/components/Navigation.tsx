"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { MessageSquare, Home, Settings, Menu, X, Activity } from "lucide-react";
import { BeanIcon } from "./icons/BeanIcon";

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  description?: string;
}

const navigationItems: NavItem[] = [
  { name: "Home", href: "/", icon: Home, description: "Home" },
  {
    name: "Chat",
    href: "/chat",
    icon: MessageSquare,
    description: "Unified LLM Interface",
  },
  {
    name: "Monitoring",
    href: "/monitoring",
    icon: Activity,
    description: "Real-time Metrics",
  },
  {
    name: "Settings",
    href: "/settings",
    icon: Settings,
    description: "API keys & Google",
  },
];

export function Navigation() {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <>
      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 h-14 bg-background/95 backdrop-blur-sm border-b flex items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2">
          <BeanIcon className="w-6 h-6 text-primary" />
          <span className="font-semibold text-foreground">BeanLLM</span>
        </Link>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="h-9 w-9 p-0"
        >
          {isMobileMenuOpen ? (
            <X className="h-5 w-5" />
          ) : (
            <Menu className="h-5 w-5" />
          )}
        </Button>
      </div>

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 bg-background/80 backdrop-blur-sm"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Mobile Menu */}
      <div
        className={cn(
          "lg:hidden fixed top-14 left-0 right-0 z-40 bg-background border-b transition-all duration-200",
          isMobileMenuOpen
            ? "opacity-100 translate-y-0"
            : "opacity-0 -translate-y-2 pointer-events-none"
        )}
      >
        <nav className="p-2 space-y-1">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                <div
                  className={cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-foreground/70 hover:bg-muted"
                  )}
                >
                  <Icon className="h-5 w-5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm">{item.name}</p>
                    {item.description && (
                      <p className="text-xs text-muted-foreground truncate">
                        {item.description}
                      </p>
                    )}
                  </div>
                </div>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Desktop Sidebar */}
      <nav
        className="hidden lg:flex w-60 border-r bg-sidebar/50 backdrop-blur-sm h-screen flex-col"
        aria-label="Side navigation"
      >
        {/* Logo */}
        <div className="p-4 border-b border-sidebar-border/50">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 flex items-center justify-center text-primary transition-transform group-hover:scale-110">
              <BeanIcon className="w-full h-full" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-sidebar-foreground">
                BeanLLM
              </h1>
              <p className="text-[10px] text-muted-foreground">Playground</p>
            </div>
          </Link>
        </div>

        {/* Nav Items */}
        <div className="flex-1 p-2 space-y-1 overflow-y-auto">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;

            return (
              <Link key={item.href} href={item.href}>
                <Button
                  variant="ghost"
                  className={cn(
                    "w-full justify-start h-auto py-2.5 px-3 text-sm font-normal transition-all",
                    "hover:bg-sidebar-accent/70",
                    isActive
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground/70"
                  )}
                >
                  <Icon
                    className={cn(
                      "h-4 w-4 mr-2.5 flex-shrink-0",
                      isActive ? "text-sidebar-primary" : "text-muted-foreground"
                    )}
                  />
                  <span className="truncate">{item.name}</span>
                </Button>
              </Link>
            );
          })}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-sidebar-border/50">
          <p className="text-[10px] text-muted-foreground text-center">
            v0.1.0 - Production Ready
          </p>
        </div>
      </nav>
    </>
  );
}

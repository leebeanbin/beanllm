"use client";

import { useState, useEffect, useCallback } from "react";

function useModalCloseBodyCleanup(open: boolean) {
  useEffect(() => {
    if (!open) {
      const t = setTimeout(() => {
        document.body.style.pointerEvents = "";
        document.body.style.overflow = "";
        document.body.style.paddingRight = "";
        document.body.removeAttribute("inert");
      }, 150);
      return () => clearTimeout(t);
    }
  }, [open]);
}
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { PasswordInput } from "@/components/ui/password-input";
import {
  HardDrive,
  FileText,
  Mail,
  Calendar,
  Table,
  Loader2,
  LogOut,
  RefreshCw,
  AlertCircle,
  ExternalLink,
  Settings,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  getGoogleServices,
  startGoogleAuth,
  getGoogleAuthStatus,
  logoutGoogle,
  saveGoogleOAuthConfig,
  type GoogleAuthStatus,
  type GoogleService,
} from "@/lib/api-client";

const SERVICE_ICONS: Record<string, React.ElementType> = {
  drive: HardDrive,
  docs: FileText,
  gmail: Mail,
  calendar: Calendar,
  sheets: Table,
};

const SERVICE_LABELS: Record<string, string> = {
  drive: "Drive",
  docs: "Docs",
  gmail: "Gmail",
  calendar: "Calendar",
  sheets: "Sheets",
};

interface GoogleConnectModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  userId?: string;
  onAuthChange?: (isAuthenticated: boolean) => void;
}

export function GoogleConnectModal({
  open,
  onOpenChange,
  userId = "default",
  onAuthChange,
}: GoogleConnectModalProps) {
  const [loading, setLoading] = useState(true);
  const [authStatus, setAuthStatus] = useState<GoogleAuthStatus | null>(null);
  const [services, setServices] = useState<GoogleService[]>([]);
  const [isConfigured, setIsConfigured] = useState(false);
  const [selectedServices, setSelectedServices] = useState<string[]>(["drive", "docs"]);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [configClientId, setConfigClientId] = useState("");
  const [configClientSecret, setConfigClientSecret] = useState("");
  const [configRedirectUri, setConfigRedirectUri] = useState("");

  const fetchStatus = useCallback(async () => {
    try {
      const [statusData, servicesData] = await Promise.all([
        getGoogleAuthStatus(userId),
        getGoogleServices(),
      ]);
      setAuthStatus(statusData);
      setServices(servicesData.services);
      setIsConfigured(servicesData.is_configured);
      onAuthChange?.(statusData.is_authenticated);
    } catch (error) {
      console.error("Failed to fetch Google auth status:", error);
    } finally {
      setLoading(false);
    }
  }, [userId, onAuthChange]);

  useModalCloseBodyCleanup(open);

  useEffect(() => {
    if (open) {
      setLoading(true);
      fetchStatus();
    }
  }, [open, fetchStatus]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const googleAuth = params.get("google_auth");
    const error = params.get("error");
    if (googleAuth === "success") {
      toast.success("Google account connected");
      fetchStatus();
      window.history.replaceState({}, "", window.location.pathname);
    } else if (error) {
      toast.error(`Google auth failed: ${error}`);
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, [fetchStatus]);

  const handleSaveConfig = async () => {
    const cid = configClientId.trim();
    const secret = configClientSecret.trim();
    const uri = configRedirectUri.trim() || "http://localhost:8000/api/auth/google/callback";
    if (!cid || !secret) {
      toast.error("Enter Client ID and Client Secret");
      return;
    }
    setSavingConfig(true);
    try {
      await saveGoogleOAuthConfig({
        client_id: cid,
        client_secret: secret,
        redirect_uri: uri || undefined,
      });
      toast.success("Google OAuth settings saved");
      setConfigClientId("");
      setConfigClientSecret("");
      setConfigRedirectUri("");
      setLoading(true);
      await fetchStatus();
    } catch (error) {
      console.error("Failed to save Google OAuth config:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to save settings"
      );
    } finally {
      setSavingConfig(false);
    }
  };

  const handleLogin = async () => {
    if (selectedServices.length === 0) {
      toast.error("Select at least one service");
      return;
    }
    setIsAuthenticating(true);
    try {
      const result = await startGoogleAuth(selectedServices, userId);
      window.location.href = result.auth_url;
    } catch (error) {
      console.error("Failed to start Google auth:", error);
      toast.error("Cannot start Google auth");
      setIsAuthenticating(false);
    }
  };

  const handleLogout = async () => {
    if (!confirm("Disconnect Google account?")) return;
    setIsLoggingOut(true);
    try {
      const result = await logoutGoogle(userId);
      if (result.success) {
        toast.success("Google account disconnected");
        setAuthStatus(null);
        onAuthChange?.(false);
      } else {
        toast.error("Logout failed");
      }
    } catch (error) {
      console.error("Failed to logout:", error);
      toast.error("Logout failed");
    } finally {
      setIsLoggingOut(false);
      fetchStatus();
    }
  };

  const toggleService = (service: string) => {
    setSelectedServices((prev) =>
      prev.includes(service) ? prev.filter((s) => s !== service) : [...prev, service]
    );
  };

  const isAuthenticated = authStatus?.is_authenticated;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          "max-w-md gap-3 border-border/40 rounded-lg antialiased",
          "max-h-[85vh] overflow-hidden flex flex-col"
        )}
      >
        <DialogHeader className="gap-1 shrink-0">
          <DialogTitle className="text-[15px] font-semibold tracking-tight">
            Google Workspace
          </DialogTitle>
          <DialogDescription className="text-[12px] text-muted-foreground">
            Connect Drive, Docs, Gmail and more to save and share chats.
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-8 shrink-0">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : !isConfigured ? (
          <div className="flex-1 overflow-y-auto space-y-4">
            <div className="rounded-lg border border-border/40 bg-muted/5 p-3 space-y-3">
              <Label className="text-[12px] font-semibold tracking-tight text-foreground flex items-center gap-1.5">
                <Settings className="h-3.5 w-3.5 text-muted-foreground" />
                Enter credentials
              </Label>
              <p className="text-[11px] text-muted-foreground">
                Create OAuth 2.0 client ID and Secret in Google Cloud Console, then enter them below.
              </p>
              <div className="space-y-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">Client ID</Label>
                  <Input
                    value={configClientId}
                    onChange={(e) => setConfigClientId(e.target.value)}
                    placeholder="xxx.apps.googleusercontent.com"
                    className={cn(
                      "h-8 text-[12px] border-border/40 transition-colors",
                      configClientId.trim() &&
                        "border-green-500/50 bg-green-500/5 focus-visible:ring-green-500/20"
                    )}
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">Client Secret</Label>
                  <PasswordInput
                    value={configClientSecret}
                    onChange={(e) => setConfigClientSecret(e.target.value)}
                    placeholder="GOCSPX-..."
                    className={cn(
                      "h-8 text-[12px] border-border/40 transition-colors",
                      configClientSecret.trim() &&
                        "border-green-500/50 bg-green-500/5 focus-visible:ring-green-500/20"
                    )}
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-muted-foreground">
                    Redirect URI (optional, default used if empty)
                  </Label>
                  <Input
                    value={configRedirectUri}
                    onChange={(e) => setConfigRedirectUri(e.target.value)}
                    placeholder="http://localhost:8000/api/auth/google/callback"
                    className={cn(
                      "h-8 text-[12px] border-border/40 font-mono transition-colors",
                      configRedirectUri.trim() &&
                        "border-green-500/50 bg-green-500/5 focus-visible:ring-green-500/20"
                    )}
                  />
                </div>
              </div>
              <Button
                size="sm"
                onClick={handleSaveConfig}
                disabled={savingConfig || !configClientId.trim() || !configClientSecret.trim()}
                className="h-8 text-[12px] font-medium"
              >
                {savingConfig ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
                ) : null}
                Save settings
              </Button>
            </div>
            <div className="flex items-start gap-2 rounded-lg border border-border/40 bg-muted/10 p-3">
              <AlertCircle className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" />
              <div className="space-y-1.5 text-[12px]">
                <p className="font-medium text-foreground">Get credentials</p>
                <p className="text-muted-foreground text-[11px]">
                  Google Cloud Console → APIs & Services → Credentials. Create an OAuth 2.0 client
                  ID and add the Redirect URI above.
                </p>
                <a
                  href="https://console.cloud.google.com/apis/credentials"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-primary hover:underline text-[12px]"
                >
                  Google Cloud Console
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
            </div>
          </div>
        ) : isAuthenticated ? (
          <div className="space-y-3 flex-1 overflow-y-auto">
            <div className="flex flex-wrap gap-2">
              {authStatus?.available_services.map((name) => {
                const Icon = SERVICE_ICONS[name] ?? FileText;
                const label = SERVICE_LABELS[name] ?? name;
                return (
                  <Tooltip key={name}>
                    <TooltipTrigger asChild>
                      <div
                        className={cn(
                          "flex h-9 w-9 items-center justify-center rounded-lg border border-border/40 bg-muted/20 text-muted-foreground",
                          "ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        )}
                      >
                        <Icon className="h-4 w-4" />
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="text-[11px]">
                      {label} connected
                    </TooltipContent>
                  </Tooltip>
                );
              })}
            </div>
            {authStatus?.expires_at && (
              <p className="text-[11px] text-muted-foreground">
                Token expires: {new Date(authStatus.expires_at).toLocaleDateString()}
              </p>
            )}
            <DialogFooter className="gap-2 sm:gap-0 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={fetchStatus}
                className="gap-1.5 h-8 text-[12px] font-medium"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Refresh
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleLogout}
                disabled={isLoggingOut}
                className="gap-1.5 h-8 text-[12px] font-medium text-destructive hover:text-destructive"
              >
                {isLoggingOut ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <LogOut className="h-3.5 w-3.5" />
                )}
                Disconnect
              </Button>
            </DialogFooter>
          </div>
        ) : (
          <div className="space-y-3 flex-1 overflow-y-auto">
            <p className="text-[12px] text-muted-foreground">
              Choose services to connect (click icons)
            </p>
            <div className="flex flex-wrap gap-2">
              {services.map((svc) => {
                const Icon = SERVICE_ICONS[svc.name] ?? FileText;
                const label = SERVICE_LABELS[svc.name] ?? svc.name;
                const isSelected = selectedServices.includes(svc.name);
                return (
                  <Tooltip key={svc.name}>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => toggleService(svc.name)}
                        className={cn(
                          "flex h-9 w-9 items-center justify-center rounded-lg border transition-colors",
                          isSelected
                            ? "border-primary bg-primary/10 text-primary ring-2 ring-primary/20"
                            : "border-border/40 bg-muted/10 text-muted-foreground hover:bg-muted/20"
                        )}
                      >
                        <Icon className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="text-[11px]">
                      {label}
                      {isSelected ? " ✓" : ""}
                    </TooltipContent>
                  </Tooltip>
                );
              })}
            </div>
            <Button
              onClick={handleLogin}
              disabled={isAuthenticating || selectedServices.length === 0}
              className="w-full h-9 text-[13px] font-medium"
            >
              {isAuthenticating ? (
                <Loader2 className="h-4 w-4 animate-spin shrink-0" />
              ) : null}
              Connect with Google
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

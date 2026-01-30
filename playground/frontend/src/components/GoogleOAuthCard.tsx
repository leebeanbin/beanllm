"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Chrome,
  Check,
  X,
  Loader2,
  LogOut,
  ExternalLink,
  FileText,
  HardDrive,
  Mail,
  Calendar,
  Table,
  RefreshCw,
  AlertCircle,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  getGoogleServices,
  startGoogleAuth,
  getGoogleAuthStatus,
  logoutGoogle,
  type GoogleAuthStatus,
  type GoogleService,
} from "@/lib/api-client";

interface GoogleOAuthCardProps {
  userId?: string;
  onAuthChange?: (isAuthenticated: boolean) => void;
}

const SERVICE_ICONS: Record<string, React.ElementType> = {
  drive: HardDrive,
  docs: FileText,
  gmail: Mail,
  calendar: Calendar,
  sheets: Table,
};

const SERVICE_LABELS: Record<string, string> = {
  drive: "Google Drive",
  docs: "Google Docs",
  gmail: "Gmail",
  calendar: "Google Calendar",
  sheets: "Google Sheets",
};

export function GoogleOAuthCard({ userId = "default", onAuthChange }: GoogleOAuthCardProps) {
  const [loading, setLoading] = useState(true);
  const [authStatus, setAuthStatus] = useState<GoogleAuthStatus | null>(null);
  const [services, setServices] = useState<GoogleService[]>([]);
  const [isConfigured, setIsConfigured] = useState(false);
  const [selectedServices, setSelectedServices] = useState<string[]>(["drive", "docs"]);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);

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

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Check for OAuth callback result in URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const googleAuth = params.get("google_auth");
    const error = params.get("error");

    if (googleAuth === "success") {
      toast.success("Google account connected");
      fetchStatus();
      // Clean URL
      window.history.replaceState({}, "", window.location.pathname);
    } else if (error) {
      toast.error(`Google auth failed: ${error}`);
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, [fetchStatus]);

  const handleLogin = async () => {
    if (selectedServices.length === 0) {
      toast.error("Select at least one service");
      return;
    }

    setIsAuthenticating(true);
    try {
      const result = await startGoogleAuth(selectedServices, userId);
      // Redirect to Google OAuth
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
      prev.includes(service)
        ? prev.filter((s) => s !== service)
        : [...prev, service]
    );
  };

  if (loading) {
    return (
      <div className="border border-border/40 rounded-lg p-6 flex items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin shrink-0" />
        <span>Loading...</span>
      </div>
    );
  }

  if (!isConfigured) {
    return (
      <div className="border border-border/40 rounded-lg overflow-hidden">
        <div className="py-3 px-3 border-b border-border/40 flex items-center gap-2">
          <Chrome className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Google Workspace</span>
        </div>
        <div className="p-3">
          <div className="flex items-start gap-3 p-3 rounded-lg border border-border/60 bg-muted/30">
            <AlertCircle className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" />
            <div className="space-y-2 text-sm">
              <p className="font-medium text-foreground">Configuration required</p>
              <p className="text-xs text-muted-foreground">
                Backend env vars: GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REDIRECT_URI
              </p>
              <a
                href="https://console.cloud.google.com/apis/credentials"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-primary hover:underline text-xs"
              >
                Configure in Google Cloud Console
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const isAuthenticated = authStatus?.is_authenticated;

  return (
    <div className="border border-border/40 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between py-3 px-3 border-b border-border/40">
        <div className="flex items-center gap-2">
          <Chrome className="h-4 w-4 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Google Workspace</span>
        </div>
        {isAuthenticated ? (
          <Badge variant="outline" className="bg-muted/50 text-foreground border-border/60">
            <Check className="h-3 w-3 mr-1 shrink-0" />
            Connected
          </Badge>
        ) : (
          <Badge variant="outline" className="text-muted-foreground">
            Not connected
          </Badge>
        )}
      </div>
      <p className="text-xs text-muted-foreground px-3 py-2 border-b border-border/40">
        Connect Drive, Docs, Gmail and more to save and share chats
      </p>

      <div className="p-3 space-y-4">
        {isAuthenticated ? (
          <>
            {/* Connected Services */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">Connected services</Label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {authStatus?.available_services.map((service) => {
                  const Icon = SERVICE_ICONS[service] || FileText;
                  return (
                    <div
                      key={service}
                      className="flex items-center gap-2 p-2 rounded-md bg-muted/30 border border-border/40"
                    >
                      <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
                      <span className="text-sm">{SERVICE_LABELS[service] || service}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Token Info */}
            {authStatus?.expires_at && (
              <div className="text-xs text-muted-foreground">
                Token expires: {new Date(authStatus.expires_at).toLocaleString()}
              </div>
            )}

            <Separator />

            {/* Actions */}
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={fetchStatus}
                className="gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleLogout}
                disabled={isLoggingOut}
                className="gap-2 text-destructive hover:text-destructive"
              >
                {isLoggingOut ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <LogOut className="h-4 w-4" />
                )}
                Disconnect
              </Button>
            </div>
          </>
        ) : (
          <>
            {/* Service Selection */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">Choose services to connect</Label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {services.map((service) => {
                  const Icon = SERVICE_ICONS[service.name] || FileText;
                  const isSelected = selectedServices.includes(service.name);

                  return (
                    <div
                      key={service.name}
                      className={cn(
                        "flex items-start gap-3 p-3 rounded-lg border border-border/40 cursor-pointer transition-colors",
                        isSelected ? "bg-muted/40 border-border/60" : "bg-background hover:bg-muted/20"
                      )}
                      onClick={() => toggleService(service.name)}
                    >
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={() => toggleService(service.name)}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium text-sm">
                            {SERVICE_LABELS[service.name] || service.name}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          {service.description}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <Separator />

            {/* Login Button */}
            <Button
              onClick={handleLogin}
              disabled={isAuthenticating || selectedServices.length === 0}
              className="w-full gap-2"
            >
              {isAuthenticating ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Chrome className="h-4 w-4" />
              )}
              Connect with Google
            </Button>

            <p className="text-xs text-muted-foreground text-center">
              Only requests access to selected services.
              You can disconnect anytime.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

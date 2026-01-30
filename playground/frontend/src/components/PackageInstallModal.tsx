"use client";

import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Loader2, Download, Check, X, AlertCircle } from "lucide-react";
import { toast } from "sonner";

interface PackageInstallModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  provider: string;
  onInstallComplete?: () => void;
}

export function PackageInstallModal({
  open,
  onOpenChange,
  provider,
  onInstallComplete,
}: PackageInstallModalProps) {
  const [installing, setInstalling] = useState(false);
  const [installStatus, setInstallStatus] = useState<{
    success: boolean;
    message: string;
    output?: string;
    error?: string;
  } | null>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    if (open) {
      setInstallStatus(null);
    }
  }, [open]);

  const handleInstall = async () => {
    setInstalling(true);
    setInstallStatus(null);

    try {
      const response = await fetch(`${apiUrl}/api/config/install-package`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          package_group: provider,
        }),
      });

      const result = await response.json();

      if (result.success) {
        setInstallStatus({
          success: true,
          message: result.message,
          output: result.output,
        });
        toast.success(`${provider} package installed`, {
          description: "Refresh the page or restart the backend.",
        });
        onInstallComplete?.();
      } else {
        setInstallStatus({
          success: false,
          message: result.message,
          error: result.error,
          output: result.output,
        });
        toast.error("Package install failed", {
          description: result.error || result.message,
        });
      }
    } catch (error) {
      console.error("Failed to install package:", error);
      setInstallStatus({
        success: false,
        message: "Error during install",
        error: error instanceof Error ? error.message : String(error),
      });
      toast.error("Package install failed", {
        description: "Check server connection.",
      });
    } finally {
      setInstalling(false);
    }
  };

  const handleClose = () => {
    if (!installing) {
      onOpenChange(false);
      setInstallStatus(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Install Provider SDK
          </DialogTitle>
          <DialogDescription>
            Installing <span className="font-medium">{provider}</span> Provider SDK.
            This may take a few minutes.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {!installStatus && (
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">
                Installing <span className="font-medium">{provider}</span> SDK enables
                that provider&apos;s models.
              </p>
            </div>
          )}

          {installStatus && (
            <div
              className={`p-4 rounded-lg ${
                installStatus.success
                  ? "bg-green-500/10 border border-green-500/30"
                  : "bg-red-500/10 border border-red-500/30"
              }`}
            >
              <div className="flex items-start gap-3">
                {installStatus.success ? (
                  <Check className="h-5 w-5 text-green-600 mt-0.5" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-red-600 mt-0.5" />
                )}
                <div className="flex-1 space-y-2">
                  <p
                    className={`text-sm font-medium ${
                      installStatus.success ? "text-green-800 dark:text-green-200" : "text-red-800 dark:text-red-200"
                    }`}
                  >
                    {installStatus.message}
                  </p>
                  {installStatus.error && (
                    <pre className="text-xs bg-black/10 dark:bg-white/10 p-2 rounded overflow-auto max-h-32">
                      {installStatus.error}
                    </pre>
                  )}
                  {installStatus.output && installStatus.success && (
                    <pre className="text-xs bg-black/10 dark:bg-white/10 p-2 rounded overflow-auto max-h-32">
                      {installStatus.output}
                    </pre>
                  )}
                </div>
              </div>
            </div>
          )}

          {installStatus?.success && (
            <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <p className="text-xs text-blue-800 dark:text-blue-200">
                💡 <strong>Next:</strong> Restart the backend or refresh the page.
              </p>
            </div>
          )}
        </div>

        <DialogFooter>
          {installStatus?.success ? (
            <Button onClick={handleClose}>Close</Button>
          ) : (
            <>
              <Button variant="outline" onClick={handleClose} disabled={installing}>
                Cancel
              </Button>
              <Button onClick={handleInstall} disabled={installing}>
                {installing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Installing...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Install
                  </>
                )}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

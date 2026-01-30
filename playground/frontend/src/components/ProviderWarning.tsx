"use client";

import { useState, useEffect } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertCircle, Download, X } from "lucide-react";
import { PackageInstallModal } from "./PackageInstallModal";

interface ProviderSDKStatus {
  provider: string;
  installed: boolean;
  package_name: string;
  install_command: string;
  error?: string | null;
}

interface ProviderSDKStatusResponse {
  providers: ProviderSDKStatus[];
  warnings: string[];
}

export function ProviderWarning() {
  const [sdkStatus, setSdkStatus] = useState<ProviderSDKStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    fetchSDKStatus();
  }, []);

  const fetchSDKStatus = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/api/config/provider-sdks`);
      if (response.ok) {
        const data = await response.json();
        setSdkStatus(data);
      }
    } catch (error) {
      console.error("Failed to fetch SDK status:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !sdkStatus) {
    return null;
  }

  const missingProviders = sdkStatus.providers.filter((p) => !p.installed);

  if (missingProviders.length === 0) {
    return null;
  }

  const handleInstallClick = (provider: string) => {
    setSelectedProvider(provider);
    setShowModal(true);
  };

  const handleInstallComplete = () => {
    setShowModal(false);
    setSelectedProvider(null);
    // SDK 상태 다시 확인
    fetchSDKStatus();
  };

  return (
    <>
      <Alert className="border-yellow-500/50 bg-yellow-500/10 mb-4">
        <AlertCircle className="h-4 w-4 text-yellow-600" />
        <AlertTitle className="text-yellow-800 dark:text-yellow-200">
          Provider SDK not installed
        </AlertTitle>
        <AlertDescription className="mt-2 space-y-2">
          <p className="text-sm text-yellow-700 dark:text-yellow-300">
            Install the following Provider SDK(s) to use these models:
          </p>
          <div className="flex flex-wrap gap-2 mt-2">
            {missingProviders.map((provider) => (
              <div
                key={provider.provider}
                className="flex items-center gap-2 bg-yellow-500/20 rounded-md px-3 py-1.5"
              >
                <span className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  {provider.provider}
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleInstallClick(provider.provider)}
                  className="h-6 px-2 text-xs bg-white dark:bg-gray-800 hover:bg-yellow-50 dark:hover:bg-yellow-900/20"
                >
                  <Download className="h-3 w-3 mr-1" />
                  Install
                </Button>
              </div>
            ))}
          </div>
          <div className="mt-3 text-xs text-yellow-600 dark:text-yellow-400">
            <p>Or install via terminal:</p>
            <code className="block mt-1 p-2 bg-yellow-500/20 rounded text-yellow-800 dark:text-yellow-200">
              {missingProviders.map((p) => p.install_command).join("\n")}
            </code>
          </div>
        </AlertDescription>
      </Alert>

      {showModal && selectedProvider && (
        <PackageInstallModal
          open={showModal}
          onOpenChange={setShowModal}
          provider={selectedProvider}
          onInstallComplete={handleInstallComplete}
        />
      )}
    </>
  );
}

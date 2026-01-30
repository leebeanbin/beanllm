"use client";

import { useState, useEffect, useRef } from "react";
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
import { PasswordInput } from "@/components/ui/password-input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Key,
  Check,
  X,
  Loader2,
  Trash2,
  AlertCircle,
  ExternalLink,
  Plus,
  FileDown,
  Upload,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface ProviderInfo {
  id: string;
  name: string;
  env_var: string;
  placeholder: string;
  description: string;
  is_configured: boolean;
  is_valid: boolean | null;
}

interface ApiKeyModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onKeysUpdated?: () => void;
}

const PROVIDER_DOCS: Record<string, string> = {
  openai: "https://platform.openai.com/api-keys",
  anthropic: "https://console.anthropic.com/settings/keys",
  google: "https://aistudio.google.com/app/apikey",
  gemini: "https://aistudio.google.com/app/apikey",
  deepseek: "https://platform.deepseek.com/api_keys",
  perplexity: "https://www.perplexity.ai/settings/api",
  tavily: "https://app.tavily.com/home",
  pinecone: "https://app.pinecone.io/",
  qdrant: "https://cloud.qdrant.io/",
};

/** env_var → provider id mapping for .env import */
const ENV_VAR_TO_PROVIDER: Record<string, string> = {
  OPENAI_API_KEY: "openai",
  ANTHROPIC_API_KEY: "anthropic",
  GOOGLE_API_KEY: "google",
  GEMINI_API_KEY: "gemini",
  DEEPSEEK_API_KEY: "deepseek",
  PERPLEXITY_API_KEY: "perplexity",
  TAVILY_API_KEY: "tavily",
  SERPAPI_API_KEY: "serpapi",
  PINECONE_API_KEY: "pinecone",
  QDRANT_API_KEY: "qdrant",
  WEAVIATE_API_KEY: "weaviate",
  NEO4J_PASSWORD: "neo4j",
};

export function ApiKeyModal({ open, onOpenChange, onKeysUpdated }: ApiKeyModalProps) {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [keyInputs, setKeyInputs] = useState<Record<string, string>>({});
  const [addProviderId, setAddProviderId] = useState<string>("");
  const [addKeyValue, setAddKeyValue] = useState("");
  const [importPreview, setImportPreview] = useState<{ provider: string; api_key: string }[] | null>(null);
  const [savingAll, setSavingAll] = useState(false);
  const [deletingProvider, setDeletingProvider] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const importFileInputRef = useRef<HTMLInputElement>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    if (open) {
      fetchProviders();
    }
  }, [open]);

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

  const fetchProviders = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/api/config/providers/all`);
      if (response.ok) {
        const data = await response.json();
        setProviders(data.providers || []);
      } else {
        toast.error("Failed to load provider list");
      }
    } catch (error) {
      console.error("Failed to fetch providers:", error);
      toast.error("Cannot connect to server");
    } finally {
      setLoading(false);
    }
  };

  /** Collect keys to save: keyInputs with value + add row (provider + key), then batch POST */
  const getKeysToSave = (): { provider: string; api_key: string }[] => {
    const entries: { provider: string; api_key: string }[] = [];
    for (const [providerId, val] of Object.entries(keyInputs)) {
      const v = (val ?? "").trim();
      if (v) entries.push({ provider: providerId, api_key: v });
    }
    if (addProviderId && addKeyValue.trim()) {
      entries.push({ provider: addProviderId, api_key: addKeyValue.trim() });
    }
    return entries;
  };

  const handleBatchSave = async () => {
    const toSave = getKeysToSave();
    if (toSave.length === 0) {
      toast.error("No keys to save");
      return;
    }
    setSavingAll(true);
    let ok = 0;
    let fail = 0;
    try {
      const results = await Promise.allSettled(
        toSave.map((e) =>
          fetch(`${apiUrl}/api/config/keys`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider: e.provider, api_key: e.api_key }),
          })
        )
      );
      for (const r of results) {
        if (r.status === "fulfilled" && r.value.ok) ok++;
        else fail++;
      }
      if (ok) {
        const savedProviders: string[] = [];
        for (let i = 0; i < toSave.length; i++) {
          const r = results[i];
          if (r.status === "fulfilled" && r.value.ok) savedProviders.push(toSave[i].provider);
        }
        for (const providerId of savedProviders) {
          try {
            await fetch(`${apiUrl}/api/config/keys/${providerId}/validate`, { method: "POST" });
          } catch {
            /* 무시, 배지만 미검증으로 유지 */
          }
        }
        toast.success(`${ok} key(s) saved` + (fail ? ` (${fail} failed)` : ""));
        setKeyInputs({});
        setAddKeyValue("");
        setAddProviderId("");
        await fetchProviders();
        onKeysUpdated?.();
      }
      if (fail && !ok) toast.error("All saves failed");
    } catch (error) {
      console.error("Batch save failed:", error);
      toast.error("An error occurred while saving");
    } finally {
      setSavingAll(false);
    }
  };

  const parseEnvText = (text: string): { provider: string; api_key: string }[] => {
    const entries: { provider: string; api_key: string }[] = [];
    const lines = text.split(/\r?\n/).filter((l) => l.trim());
    for (const line of lines) {
      const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
      if (!m) continue;
      const envVar = m[1];
      const value = m[2].replace(/^["']|["']$/g, "").trim();
      const providerId = ENV_VAR_TO_PROVIDER[envVar];
      if (providerId && value) entries.push({ provider: providerId, api_key: value });
    }
    return entries;
  };

  const handleImportFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      const entries = parseEnvText(text);
      setImportPreview(entries);
      if (entries.length === 0) toast.error("No keys recognized. Check .env format (KEY=value).");
    };
    reader.readAsText(file, "UTF-8");
  };

  const handleImportConfirm = async () => {
    if (!importPreview || importPreview.length === 0) return;
    setImporting(true);
    let ok = 0;
    let fail = 0;
    try {
      for (const e of importPreview) {
        try {
          const res = await fetch(`${apiUrl}/api/config/keys`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider: e.provider, api_key: e.api_key }),
          });
          if (res.ok) ok++;
          else fail++;
        } catch {
          fail++;
        }
      }
      if (ok) {
        toast.success(`${ok} key(s) saved` + (fail ? ` (${fail} failed)` : ""));
        setImportPreview(null);
        await fetchProviders();
        onKeysUpdated?.();
      }
      if (fail && !ok) toast.error("All saves failed");
    } finally {
      setImporting(false);
    }
  };

  const handleImportCancel = () => {
    setImportPreview(null);
  };

  const handleDeleteKey = async (providerId: string) => {
    if (!confirm(`Delete ${providerId} key?`)) return;
    setDeletingProvider(providerId);
    try {
      const response = await fetch(`${apiUrl}/api/config/keys/${providerId}`, {
        method: "DELETE",
      });
      if (response.ok) {
        toast.success(`${providerId} key deleted`);
        await fetchProviders();
        onKeysUpdated?.();
      } else {
        toast.error("Delete failed");
      }
    } catch (error) {
      console.error("Failed to delete API key:", error);
      toast.error("Delete failed");
    } finally {
      setDeletingProvider(null);
    }
  };

  const llmProviders = providers.filter((p) =>
    ["openai", "anthropic", "google", "gemini", "deepseek", "perplexity"].includes(p.id)
  );
  const otherProviders = providers.filter(
    (p) => !["openai", "anthropic", "google", "gemini", "deepseek", "perplexity"].includes(p.id)
  );

  const renderProvider = (provider: ProviderInfo) => {
    const isConfigured = provider.is_configured;
    const isValid = provider.is_valid;
    const isDeleting = deletingProvider === provider.id;
    const docUrl = PROVIDER_DOCS[provider.id];
    const keyVal = keyInputs[provider.id] ?? "";
    const inputValid = keyVal.trim().length > 0;

    return (
      <div
        key={provider.id}
        className={cn(
          "rounded-lg border border-border/40 p-3 space-y-2",
          isConfigured && isValid && "bg-muted/10 border-border/60",
          isConfigured && isValid === false && "border-destructive/40 bg-destructive/5",
          isConfigured && isValid === null && "bg-muted/10 border-border/60"
        )}
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Key className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <span className="text-[13px] font-semibold tracking-tight text-foreground truncate">
              {provider.name}
            </span>
            {docUrl && (
              <a
                href={docUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-foreground transition-colors shrink-0"
                aria-label="Open docs"
              >
                <ExternalLink className="h-3 w-3" />
              </a>
            )}
          </div>
          <div className="shrink-0">
            {isConfigured ? (
              <>
                {isValid === true && (
                  <Badge
                    variant="outline"
                    className="bg-green-500/10 text-green-600 border-green-500/30 text-[11px] font-medium"
                  >
                    <Check className="h-2.5 w-2.5 mr-0.5" />
                    Valid
                  </Badge>
                )}
                {isValid === false && (
                  <Badge
                    variant="outline"
                    className="bg-red-500/10 text-red-600 border-red-500/30 text-[11px] font-medium"
                  >
                    <X className="h-2.5 w-2.5 mr-0.5" />
                    Invalid
                  </Badge>
                )}
                {isValid === null && (
                  <Badge
                    variant="outline"
                    className="bg-amber-500/10 text-amber-600 border-amber-500/30 text-[11px] font-medium"
                  >
                    <AlertCircle className="h-2.5 w-2.5 mr-0.5" />
                    Unverified
                  </Badge>
                )}
              </>
            ) : (
              <Badge variant="outline" className="text-muted-foreground text-[11px] font-medium">
                Not set
              </Badge>
            )}
          </div>
        </div>
        <p className="text-[11px] text-muted-foreground leading-snug">{provider.description}</p>
        <div className="flex items-center gap-2">
          <PasswordInput
            placeholder={provider.placeholder || "Enter key…"}
            value={keyVal}
            onChange={(e) =>
              setKeyInputs((prev) => ({ ...prev, [provider.id]: e.target.value }))
            }
            className={cn(
              "flex-1 h-8 text-[12px] border-border/40 transition-colors",
              inputValid && "border-green-500/50 bg-green-500/5 focus-visible:ring-green-500/20"
            )}
          />
        </div>
        {isConfigured && (
          <div className="flex items-center gap-1.5 pt-1">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleDeleteKey(provider.id)}
              disabled={isDeleting}
              className="h-7 text-[11px] font-medium text-destructive hover:text-destructive gap-1"
            >
              {isDeleting ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Trash2 className="h-3 w-3" />
              )}
              Delete
            </Button>
          </div>
        )}
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          "max-w-2xl max-h-[85vh] overflow-hidden flex flex-col gap-0",
          "border-border/40 rounded-lg antialiased"
        )}
      >
        <DialogHeader className="gap-1">
          <DialogTitle className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
            <Key className="h-4 w-4 text-muted-foreground" />
            API Keys
          </DialogTitle>
          <DialogDescription className="text-[12px] text-muted-foreground">
            Enter values and click Save to save and validate at once. Valid fields are highlighted in green.
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto space-y-4 py-4">
            {/* Add key — type and include in save */}
            <section className="rounded-lg border border-border/40 bg-muted/5 p-3 space-y-2">
              <Label className="text-[12px] font-semibold tracking-tight text-foreground flex items-center gap-1.5">
                <Plus className="h-3.5 w-3.5 text-muted-foreground" />
                Add key
              </Label>
              <div className="flex flex-wrap items-end gap-2">
                <div className="flex-1 min-w-[140px] space-y-1">
                  <span className="text-[11px] text-muted-foreground">Provider</span>
                  <Select
                    value={addProviderId || undefined}
                    onValueChange={setAddProviderId}
                  >
                    <SelectTrigger className="h-8 text-[12px] border-border/40">
                      <SelectValue placeholder="Select" />
                    </SelectTrigger>
                    <SelectContent>
                      {providers.map((p) => (
                        <SelectItem key={p.id} value={p.id} className="text-[12px]">
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex-[2] min-w-[180px] space-y-1">
                  <span className="text-[11px] text-muted-foreground">Key value</span>
                  <PasswordInput
                    placeholder="Enter key to include in save"
                    value={addKeyValue}
                    onChange={(e) => setAddKeyValue(e.target.value)}
                    className={cn(
                      "h-8 text-[12px] border-border/40 transition-colors",
                      addKeyValue.trim() &&
                        "border-green-500/50 bg-green-500/5 focus-visible:ring-green-500/20"
                    )}
                  />
                </div>
              </div>
            </section>

            {/* Import: .env file → preview → save all */}
            <section className="rounded-lg border border-border/40 bg-muted/5 p-3 space-y-2">
              <Label className="text-[12px] font-semibold tracking-tight text-foreground flex items-center gap-1.5">
                <FileDown className="h-3.5 w-3.5 text-muted-foreground" />
                Import (.env file)
              </Label>
              <p className="text-[11px] text-muted-foreground">
                Load a .env file to auto-fill recognized keys. You can save them all at once after confirming.
              </p>
              <input
                ref={importFileInputRef}
                type="file"
                accept=".env,.env.*,*.env,text/plain"
                className="hidden"
                onChange={handleImportFileSelect}
                aria-label="Select .env file"
              />
              {importPreview === null ? (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => importFileInputRef.current?.click()}
                  className="h-8 text-[12px] font-medium gap-1.5"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Load file
                </Button>
              ) : importPreview.length === 0 ? (
                <div className="text-[12px] text-muted-foreground flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 shrink-0" />
                  No keys recognized. Ensure format like OPENAI_API_KEY=sk-…
                  <Button type="button" variant="ghost" size="sm" onClick={handleImportCancel} className="h-7 text-[11px]">
                    Choose again
                  </Button>
                </div>
              ) : (
                <div className="space-y-2">
                  <p className="text-[12px] text-foreground font-medium">
                    The following {importPreview.length} key(s) will be saved:
                  </p>
                  <p className="text-[11px] text-muted-foreground break-words">
                    {importPreview
                      .map((e) => (providers.find((p) => p.id === e.provider)?.name ?? e.provider))
                      .join(", ")}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      size="sm"
                      onClick={handleImportConfirm}
                      disabled={importing}
                      className="h-8 text-[12px] font-medium"
                    >
                      {importing ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
                      ) : null}
                      Save all
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={handleImportCancel}
                      disabled={importing}
                      className="h-8 text-[12px] font-medium"
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
            </section>

            {/* Keys by provider */}
            <section className="space-y-2">
              <Label className="text-[12px] font-semibold tracking-tight text-foreground">
                Keys by provider
              </Label>
              <div className="grid gap-2 sm:grid-cols-2">
                {llmProviders.map(renderProvider)}
                {otherProviders.map(renderProvider)}
              </div>
            </section>
          </div>
        )}

        <DialogFooter className="border-t border-border/40 pt-4 shrink-0 gap-2 sm:gap-2">
          <Button
            size="sm"
            onClick={handleBatchSave}
            disabled={savingAll || getKeysToSave().length === 0}
            className="h-8 text-[12px] font-medium"
          >
            {savingAll ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
            ) : null}
            Save
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onOpenChange(false)}
            className="h-8 text-[12px] font-medium"
          >
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function useApiKeyCheck() {
  const [missingProviders, setMissingProviders] = useState<string[]>([]);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const checkProvider = async (providerId: string): Promise<boolean> => {
    try {
      const response = await fetch(`${apiUrl}/api/config/keys/${providerId}`);
      return response.ok;
    } catch {
      return false;
    }
  };

  const checkRequiredProviders = async (requiredProviders: string[]) => {
    const missing: string[] = [];
    for (const provider of requiredProviders) {
      const hasKey = await checkProvider(provider);
      if (!hasKey) missing.push(provider);
    }
    setMissingProviders(missing);
    return missing.length === 0;
  };

  return { missingProviders, checkRequiredProviders };
}

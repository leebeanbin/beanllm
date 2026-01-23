"use client";

import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { LoaderCircle, Settings, X } from "lucide-react";
import { ParameterTooltip } from "@/components/ParameterTooltip";

interface ModelParameters {
  supports: {
    temperature: boolean;
    max_tokens: boolean;
    top_p: boolean;
    frequency_penalty: boolean;
    presence_penalty: boolean;
  };
  max_tokens: number;
  default_temperature: number;
}

interface ModelSettingsPanelProps {
  model: string;
  temperature: number;
  maxTokens: number;
  topP: number;
  frequencyPenalty: number;
  presencePenalty: number;
  onTemperatureChange: (value: number) => void;
  onMaxTokensChange: (value: number) => void;
  onTopPChange: (value: number) => void;
  onFrequencyPenaltyChange: (value: number) => void;
  onPresencePenaltyChange: (value: number) => void;
  showSettings: boolean;
  onToggleSettings: () => void;
}

export function ModelSettingsPanel({
  model,
  temperature,
  maxTokens,
  topP,
  frequencyPenalty,
  presencePenalty,
  onTemperatureChange,
  onMaxTokensChange,
  onTopPChange,
  onFrequencyPenaltyChange,
  onPresencePenaltyChange,
  showSettings,
  onToggleSettings,
}: ModelSettingsPanelProps) {
  const [modelParams, setModelParams] = useState<ModelParameters | null>(null);

  useEffect(() => {
    const fetchModelParameters = async () => {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      try {
        const response = await fetch(`${apiUrl}/api/models/${encodeURIComponent(model)}/parameters`);
        if (response.ok) {
          const data = await response.json();
          setModelParams(data);
          if (data.default_temperature !== undefined) {
            onTemperatureChange(data.default_temperature);
          }
          if (data.max_tokens) {
            onMaxTokensChange(Math.min(maxTokens, data.max_tokens));
          }
        }
      } catch (error) {
        setModelParams({
          supports: {
            temperature: true,
            max_tokens: true,
            top_p: true,
            frequency_penalty: true,
            presence_penalty: true,
          },
          max_tokens: 4000,
          default_temperature: 0.7,
        });
      }
    };
    fetchModelParameters();
  }, [model, maxTokens, onTemperatureChange, onMaxTokensChange]);

  if (!showSettings) return null;

  return (
    <Card className="mb-4 border-border/50 bg-card/80 backdrop-blur-sm shadow-sm">
      <CardContent className="pt-5 pb-5">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Settings</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Adjust parameters for {model}
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleSettings}
            className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-accent/50"
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>

        {!modelParams ? (
          <div className="flex items-center justify-center py-8">
            <LoaderCircle className="h-5 w-5 animate-spin text-muted-foreground" />
            <span className="ml-2 text-sm text-muted-foreground">Loading parameters...</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {modelParams.supports.temperature && (
              <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                <ParameterTooltip parameter="temperature">
                  <Label htmlFor="temperature" className="text-sm font-medium text-foreground">
                    Temperature
                  </Label>
                </ParameterTooltip>
                <div className="flex items-center justify-end">
                  <span className="text-xs font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded-md">
                    {temperature.toFixed(1)}
                  </span>
                </div>
                <Input
                  id="temperature"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
                  className="h-1.5 accent-primary/60"
                />
              </div>
            )}

            {modelParams.supports.max_tokens && (
              <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                <ParameterTooltip parameter="max_tokens">
                  <Label htmlFor="max-tokens" className="text-sm font-medium text-foreground">
                    Max Tokens
                  </Label>
                </ParameterTooltip>
                <div className="flex items-center justify-between">
                  {modelParams.max_tokens && (
                    <span className="text-xs text-muted-foreground">
                      Max: {modelParams.max_tokens.toLocaleString()}
                    </span>
                  )}
                </div>
                <Input
                  id="max-tokens"
                  type="number"
                  min="1"
                  max={modelParams.max_tokens || 4000}
                  value={maxTokens}
                  onChange={(e) => onMaxTokensChange(parseInt(e.target.value) || 1000)}
                  className="h-9 bg-background/50 border-border/50 focus:border-primary/50"
                />
              </div>
            )}

            {modelParams.supports.top_p && (
              <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                <ParameterTooltip parameter="top_p">
                  <Label htmlFor="top-p" className="text-sm font-medium text-foreground">
                    Top P
                  </Label>
                </ParameterTooltip>
                <div className="flex items-center justify-end">
                  <span className="text-xs font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded-md">
                    {topP.toFixed(2)}
                  </span>
                </div>
                <Input
                  id="top-p"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={topP}
                  onChange={(e) => onTopPChange(parseFloat(e.target.value))}
                  className="h-1.5 accent-primary/60"
                />
              </div>
            )}

            {modelParams.supports.frequency_penalty && (
              <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                <ParameterTooltip parameter="frequency_penalty">
                  <Label htmlFor="frequency-penalty" className="text-sm font-medium text-foreground">
                    Frequency Penalty
                  </Label>
                </ParameterTooltip>
                <div className="flex items-center justify-end">
                  <span className="text-xs font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded-md">
                    {frequencyPenalty.toFixed(1)}
                  </span>
                </div>
                <Input
                  id="frequency-penalty"
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={frequencyPenalty}
                  onChange={(e) => onFrequencyPenaltyChange(parseFloat(e.target.value))}
                  className="h-1.5 accent-primary/60"
                />
              </div>
            )}

            {modelParams.supports.presence_penalty && (
              <div className="space-y-2.5 p-4 rounded-lg bg-muted/30 border border-border/30">
                <ParameterTooltip parameter="presence_penalty">
                  <Label htmlFor="presence-penalty" className="text-sm font-medium text-foreground">
                    Presence Penalty
                  </Label>
                </ParameterTooltip>
                <div className="flex items-center justify-end">
                  <span className="text-xs font-medium text-primary/80 bg-primary/10 px-2 py-0.5 rounded-md">
                    {presencePenalty.toFixed(1)}
                  </span>
                </div>
                <Input
                  id="presence-penalty"
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={presencePenalty}
                  onChange={(e) => onPresencePenaltyChange(parseFloat(e.target.value))}
                  className="h-1.5 accent-primary/60"
                />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

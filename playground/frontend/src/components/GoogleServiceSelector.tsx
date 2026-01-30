"use client";

import { GoogleService } from "@/types/chat";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import {
  FileText,
  FolderOpen,
  Mail,
  Calendar,
  Table,
} from "lucide-react";

interface GoogleServiceSelectorProps {
  selectedServices: GoogleService[];
  onChange: (services: GoogleService[]) => void;
  className?: string;
}

const GOOGLE_SERVICES: Array<{
  value: GoogleService;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}> = [
  {
    value: "docs",
    label: "Google Docs",
    icon: FileText,
  },
  {
    value: "drive",
    label: "Google Drive",
    icon: FolderOpen,
  },
  {
    value: "gmail",
    label: "Gmail",
    icon: Mail,
  },
  {
    value: "calendar",
    label: "Google Calendar",
    icon: Calendar,
  },
  {
    value: "sheets",
    label: "Google Sheets",
    icon: Table,
  },
];

export function GoogleServiceSelector({
  selectedServices,
  onChange,
  className,
}: GoogleServiceSelectorProps) {
  const toggleService = (service: GoogleService) => {
    if (selectedServices.includes(service)) {
      onChange(selectedServices.filter((s) => s !== service));
    } else {
      onChange([...selectedServices, service]);
    }
  };

  return (
    <div className={cn("border border-border/40 rounded-lg overflow-hidden", className)}>
      <div className="py-3 px-3 border-b border-border/40">
        <span className="text-sm font-medium text-foreground">Select Google Services</span>
      </div>
      <div className="p-3 grid grid-cols-2 gap-3">
        {GOOGLE_SERVICES.map((service) => {
          const Icon = service.icon;
          const isChecked = selectedServices.includes(service.value);
          return (
            <div key={service.value} className="flex items-center space-x-2">
              <Checkbox
                id={`google-${service.value}`}
                checked={isChecked}
                onCheckedChange={() => toggleService(service.value)}
              />
              <Label htmlFor={`google-${service.value}`} className="flex items-center gap-2 cursor-pointer">
                <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
                <span className="text-sm">{service.label}</span>
              </Label>
            </div>
          );
        })}
      </div>
    </div>
  );
}

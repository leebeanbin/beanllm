"use client";

/**
 * Chat & AI Metrics — 모니터링 대시보드
 *
 * ChatGPT-style 토큰 원칙 (Minimal·System-first, 타이포 위계, accent만 강조):
 * - bg.canvas/surface/elevated → background/card/muted (globals 시맨틱 map)
 * - text.primary/secondary → foreground / muted-foreground
 * - accent(primary) → CTA·선택·Healthy 등 강조만, 카드 배경에는 미사용
 * - state.danger → destructive (에러/Disconnected)
 * - radius-card, border.subtle(border/30), spacing 일관
 */

import { useId, useMemo, useState, useEffect, useCallback } from "react";
import { PageLayout } from "@/components/PageLayout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { API_URL } from "@/lib/api-client";
import { cn } from "@/lib/utils";
import {
  RefreshCw,
  Activity,
  AlertTriangle,
  Clock,
  Zap,
  Server,
  Database,
  MessageSquare,
  TrendingUp,
  TrendingDown,
  ChevronRight,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

/**
 * ChatGPT-style token 원칙: Minimal·System-first, accent는 강조에만.
 * 차트는 데이터용 --chart-* (배경에 브랜드 색 사용 안 함).
 */
const CHART_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];

/** Minimal·System-first: 카드 = 배경 톤 + border.subtle만 (accent 없음) */
const CARD_CLASS = "bg-background border border-border/20 rounded-lg overflow-hidden";
const CARD_HEADER_BORDER = "border-b border-border/20";

// ===========================================
// Types
// ===========================================

interface MetricsSummary {
  total_requests: number;
  total_errors: number;
  error_rate: number;
  avg_response_time_ms: number;
  min_response_time_ms: number;
  max_response_time_ms: number;
  p50_response_time_ms: number;
  p95_response_time_ms: number;
  p99_response_time_ms: number;
  last_updated: string;
}

interface RequestTrend {
  minute: number;
  requests: number;
  errors: number;
}

interface EndpointStats {
  endpoint: string;
  method: string;
  count: number;
  errors: number;
  avg_time_ms: number;
  error_rate: number;
}

interface TokenUsage {
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  request_count: number;
  avg_tokens_per_request: number;
}

interface SystemHealth {
  status: string;
  redis_connected: boolean;
  kafka_connected: boolean;
  uptime_seconds: number;
  timestamp: string;
}

interface DashboardData {
  summary: MetricsSummary;
  request_trend: RequestTrend[];
  top_endpoints: EndpointStats[];
  token_usage: TokenUsage[];
  health: SystemHealth;
}

/** 챗 1회 단위 상세 이력 (CHAT_HISTORY_METRICS.md) */
interface ChatHistoryItem {
  request_id: string;
  model: string;
  at_ts: number;
  at_minute: number;
  request_preview: string;
  response_preview: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  duration_ms: number;
}

// ===========================================
// Helper Functions
// ===========================================

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(1)}M`;
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1)}K`;
  }
  return num.toString();
}

/** Minimal·System-first: 메서드 배지는 중립(border.subtle), accent 없음. 필요 시 CTA만 primary */
function getMethodClass(method: string): string {
  return "border border-border/50 text-muted-foreground bg-transparent";
}

// ===========================================
// Components
// ===========================================

/** ChatGPT-style: bg.surface, typography 위계, accent 없음(값은 text.primary) */
function StatCard({
  title,
  value,
  unit,
  icon: Icon,
  trend,
  className,
}: {
  title: string;
  value: string | number;
  unit?: string;
  icon: React.ComponentType<{ className?: string }>;
  trend?: "up" | "down" | null;
  className?: string;
}) {
  return (
    <div
      className={cn(
        [CARD_CLASS, "p-4"].join(" "),
        "transition-[box-shadow] hover:shadow-[var(--elevation-1)]",
        className
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-xs text-muted-foreground truncate font-medium">{title}</p>
          <div className="flex items-baseline gap-1.5 mt-1">
            <p className="text-xl sm:text-2xl font-semibold truncate text-foreground">{value}</p>
            {unit && <span className="text-xs text-muted-foreground">{unit}</span>}
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {trend && (trend === "up" ? <TrendingUp className="w-4 h-4 text-muted-foreground" /> : <TrendingDown className="w-4 h-4 text-muted-foreground" />)}
          <div className="w-8 h-8 rounded-[var(--radius-button)] bg-muted flex items-center justify-center">
            <Icon className="w-4 h-4 text-muted-foreground" />
          </div>
        </div>
      </div>
    </div>
  );
}

/** ChatGPT-style: accent = status dot·Connected만, danger = Disconnected. bg.surface·border.subtle */
function HealthIndicator({ health }: { health: SystemHealth }) {
  const statusConfig = {
    healthy: { dot: "bg-primary", text: "text-foreground" },
    degraded: { dot: "bg-muted-foreground/60", text: "text-muted-foreground" },
    unhealthy: { dot: "bg-destructive", text: "text-destructive" },
  };
  const config = statusConfig[health.status as keyof typeof statusConfig] || statusConfig.unhealthy;

  return (
    <div className={CARD_CLASS}>
      <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
        <Server className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">System Health</span>
      </div>
      <div className="p-4 space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Status</span>
          <div className="flex items-center gap-2">
            <span className={cn("w-2 h-2 rounded-full animate-pulse", config.dot)} />
            <span className={cn("font-medium", config.text)}>{health.status}</span>
          </div>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Redis</span>
          <Badge
            variant="outline"
            className={cn(
              "text-xs font-mono border-border/50",
              health.redis_connected ? "text-foreground bg-transparent" : "text-destructive border-destructive/40 bg-destructive/5"
            )}
          >
            {health.redis_connected ? "Connected" : "Disconnected"}
          </Badge>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Kafka</span>
          <Badge
            variant="outline"
            className={cn(
              "text-xs font-mono border-border/50",
              health.kafka_connected ? "text-foreground bg-transparent" : "text-destructive border-destructive/40 bg-destructive/5"
            )}
          >
            {health.kafka_connected ? "Connected" : "Disconnected"}
          </Badge>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Uptime</span>
          <span className="font-medium text-foreground tabular-nums">{formatUptime(health.uptime_seconds)}</span>
        </div>
      </div>
    </div>
  );
}

/** Request Trend — AreaChart(요청) + Line(에러). Recharts는 퍼센트 높이 시 flex에서 0이 되는 경우 많음 → 픽셀 높이 고정 + useId로 gradient id 충돌 방지 */
const REQUEST_TREND_CHART_HEIGHT = 200;

function RequestTrendChart({ data }: { data: RequestTrend[] }) {
  const safeId = useId().replace(/:/g, "-");
  const chartData = useMemo(
    () =>
      data.slice(-60).map((d) => ({
        time: new Date(d.minute * 1000).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }),
        requests: d.requests,
        errors: d.errors,
      })),
    [data]
  );

  return (
    <div className={cn(CARD_CLASS, "min-h-[260px]")}>
      <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
        <Activity className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">Request Trend · 챗 전용 (Last 60 min)</span>
      </div>
      <div className="p-4 flex flex-col" role="img" aria-label="요청·에러 추세 차트">
        {chartData.length === 0 ? (
          <div className="flex items-center justify-center text-muted-foreground text-sm" style={{ height: REQUEST_TREND_CHART_HEIGHT }}>
            No data available
          </div>
        ) : (
          <div style={{ width: "100%", height: REQUEST_TREND_CHART_HEIGHT }}>
            <ResponsiveContainer width="100%" height={REQUEST_TREND_CHART_HEIGHT}>
            <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id={safeId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border/30" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
                tickLine={false}
                axisLine={{ stroke: "var(--border)" }}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
                tickLine={false}
                axisLine={false}
                allowDecimals={false}
                width={28}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--card)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-md)",
                  fontSize: 12,
                }}
                labelStyle={{ fontFamily: "var(--font-family-base)" }}
                formatter={(value: number, name: string) => [value, name]}
                labelFormatter={(_, payload) =>
                  payload?.[0]?.payload?.time ? `시간 ${payload[0].payload.time}` : ""
                }
              />
              <Area
                type="monotone"
                dataKey="requests"
                name="요청"
                stroke="var(--chart-1)"
                fill={`url(#${safeId})`}
                strokeWidth={1.5}
              />
              <Line
                type="monotone"
                dataKey="errors"
                name="에러"
                stroke="var(--destructive)"
                strokeWidth={1.5}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

function EndpointTable({ endpoints }: { endpoints: EndpointStats[] }) {
  return (
    <div className={CARD_CLASS}>
      <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
        <Zap className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">Top Endpoints · 챗(AI) 구간</span>
      </div>
      <div className="p-4">
        {endpoints.length === 0 ? (
          <div className="py-8 text-center text-muted-foreground text-sm">
            No endpoint data available
          </div>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {endpoints.map((endpoint, index) => (
              <div
                key={index}
                className="flex flex-col sm:flex-row sm:items-center gap-2 p-3 rounded-[var(--radius-button)] bg-muted/20 hover:bg-muted/40 transition-colors"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <Badge
                    variant="outline"
                    className={cn("text-xs flex-shrink-0 font-mono", getMethodClass(endpoint.method))}
                  >
                    {endpoint.method}
                  </Badge>
                  <span className="text-xs sm:text-sm font-mono truncate">
                    {endpoint.endpoint}
                  </span>
                </div>
                <div className="flex items-center gap-2 sm:gap-4 text-xs text-muted-foreground">
                  <span className="flex-shrink-0" title="요청 수">{formatNumber(endpoint.count)} reqs</span>
                  <span className="flex-shrink-0" title="에러 수">{formatNumber(endpoint.errors)} err</span>
                  <span className="flex-shrink-0" title="평균 응답시간">{endpoint.avg_time_ms.toFixed(0)}ms</span>
                  {endpoint.error_rate > 0 && (
                    <span className="text-destructive flex-shrink-0 font-medium" title="에러율">
                      {endpoint.error_rate.toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** Token Usage PieChart — 모델별 토큰 비중. Technical Elegance: 데이터 중심, 프로젝트 --chart-* */
function TokenUsagePieChart({ usage }: { usage: TokenUsage[] }) {
  const chartData = useMemo(
    () =>
      usage.map((u) => ({
        name: u.model || "unknown",
        value: u.total_tokens,
        count: u.request_count,
      })),
    [usage]
  );

  if (chartData.length === 0) {
    return (
      <div className={CARD_CLASS}>
        <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
          <MessageSquare className="w-4 h-4 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Token 비중 · 모델별 (Pie)</span>
        </div>
        <div className="h-48 flex items-center justify-center text-muted-foreground text-sm font-mono">
          No token data
        </div>
      </div>
    );
  }

  return (
    <div className={CARD_CLASS}>
      <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
        <MessageSquare className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">Token 비중 · 모델별 (Pie)</span>
      </div>
      <div className="p-4">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <Pie
              data={chartData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={48}
              outerRadius={80}
              paddingAngle={1}
              stroke="var(--border)"
            >
              {chartData.map((_, i) => (
                <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                fontSize: 12,
              }}
              formatter={(value: number, name: string, item: { payload?: { count?: number } }) => [
                `${formatNumber(value)} (${item?.payload?.count ?? 0} req)`,
                name,
              ]}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** Response Time BarChart — Min/P50/P95/P99/Max. Technical Elegance: 정밀함·정보 계층 */
function ResponseTimeBarChart({ summary }: { summary: MetricsSummary }) {
  const chartData = useMemo(
    () => [
      { label: "Min", value: summary.min_response_time_ms, fill: "var(--chart-1)" },
      { label: "P50", value: summary.p50_response_time_ms, fill: "var(--chart-2)" },
      { label: "P95", value: summary.p95_response_time_ms, fill: "var(--chart-3)" },
      { label: "P99", value: summary.p99_response_time_ms, fill: "var(--chart-4)" },
      { label: "Max", value: summary.max_response_time_ms, fill: "var(--chart-5)" },
    ],
    [summary]
  );

  return (
    <div className={CARD_CLASS}>
      <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
        <Database className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">Response Time 분포 · Bar (ms)</span>
      </div>
      <div className="p-4">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fontFamily: "var(--font-family-base)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
            />
            <YAxis
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
              width={32}
            />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                fontSize: 12,
              }}
              formatter={(value: number) => [`${value} ms`, "응답시간"]}
            />
            <Bar dataKey="value" name="ms" radius={[4, 4, 0, 0]}>
              {chartData.map((d, i) => (
                <Cell key={i} fill={d.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** 챗 이력 시간대별 분포 BarChart — 어느 시간대에 많이 했는지 */
function ChatHistoryTimeChart({ items }: { items: ChatHistoryItem[] }) {
  const chartData = useMemo(() => {
    const byMinute: Record<number, number> = {};
    items.forEach((i) => {
      byMinute[i.at_minute] = (byMinute[i.at_minute] ?? 0) + 1;
    });
    return Object.entries(byMinute)
      .map(([k, v]) => ({
        minute: Number(k),
        time: new Date(Number(k) * 60 * 1000).toLocaleTimeString("ko-KR", {
          hour: "2-digit",
          minute: "2-digit",
        }),
        count: v,
      }))
      .sort((a, b) => a.minute - b.minute)
      .slice(-24);
  }, [items]);

  if (chartData.length === 0) {
    return (
      <div className={CARD_CLASS}>
        <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
          <Clock className="w-4 h-4 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">챗 이력 · 시간대별 분포 (Bar)</span>
        </div>
        <div className="h-40 flex items-center justify-center text-muted-foreground text-sm font-mono">
          챗 이력이 없거나 구간 내 데이터 없음
        </div>
      </div>
    );
  }

  return (
    <div className={CARD_CLASS}>
      <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
        <Clock className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">챗 이력 · 시간대별 분포 (Bar)</span>
      </div>
      <div className="p-4">
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              interval="preserveStartEnd"
            />
            <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={24} allowDecimals={false} />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                fontSize: 12,
              }}
              formatter={(value: number) => [value, "챗 수"]}
              labelFormatter={(_, payload) =>
                payload?.[0]?.payload?.time ? payload[0].payload.time : ""
              }
            />
            <Bar dataKey="count" name="챗 수" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** ChatGPT-style: typography 위계, accent 없음(숫자는 text.primary/secondary) */
function TokenUsageTable({ usage }: { usage: TokenUsage[] }) {
  return (
    <div className={CARD_CLASS}>
      <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
        <MessageSquare className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">Token Usage by Model · 챗(AI) 세부</span>
      </div>
      <div className="p-4">
        {usage.length === 0 ? (
          <div className="py-8 text-center text-muted-foreground text-sm">
            No token usage data available
          </div>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {usage.map((item, index) => (
              <div
                key={index}
                className="p-3 rounded-[var(--radius-button)] bg-muted/20 hover:bg-muted/40 transition-colors"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs sm:text-sm font-medium truncate max-w-[60%] text-foreground">
                    {item.model}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {item.request_count} requests
                    {item.request_count > 0 && item.avg_tokens_per_request > 0 && (
                      <span className="ml-1">· avg {formatNumber(Math.round(item.avg_tokens_per_request))}/req</span>
                    )}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-xs flex-wrap text-muted-foreground tabular-nums">
                  <span title="입력 토큰">In: {formatNumber(item.input_tokens)}</span>
                  <span title="출력 토큰">Out: {formatNumber(item.output_tokens)}</span>
                  <span className="font-medium text-foreground" title="총 토큰">
                    Total: {formatNumber(item.total_tokens)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** ChatGPT-style: bg.surface, border.subtle, accent 없음. elevated = row hover */
function ChatHistoryTable({
  items,
  onSelectItem,
}: {
  items: ChatHistoryItem[];
  onSelectItem?: (item: ChatHistoryItem) => void;
}) {
  return (
    <div className={CARD_CLASS}>
      <div className={cn("py-3 px-4 flex items-center gap-2", CARD_HEADER_BORDER)}>
        <MessageSquare className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">
          챗 상세 이력 · 평시 수집 · 행 클릭 시 더 자세히 보기
        </span>
      </div>
      <div className="p-4 overflow-x-auto">
        {items.length === 0 ? (
          <div className="py-8 text-center text-muted-foreground text-sm">
            챗 이력이 없습니다. 챗 요청이 발생하면 이곳에 표시됩니다.
          </div>
        ) : (
          <div className="max-h-80 overflow-y-auto">
            <table className="w-full text-xs sm:text-sm border-collapse">
              <thead className="sticky top-0 bg-muted/60 z-10">
                <tr className="border-b border-border/30">
                  <th className="text-left py-2 px-2 font-medium text-muted-foreground">시간</th>
                  <th className="text-left py-2 px-2 font-medium text-muted-foreground">모델</th>
                  <th className="text-left py-2 px-2 font-medium text-muted-foreground max-w-[12ch] truncate" title="요청 요약">요청</th>
                  <th className="text-left py-2 px-2 font-medium text-muted-foreground max-w-[12ch] truncate" title="응답 요약">응답</th>
                  <th className="text-right py-2 px-2 font-medium text-muted-foreground">토큰</th>
                  <th className="text-right py-2 px-2 font-medium text-muted-foreground">소요</th>
                  {onSelectItem && <th className="w-8" />}
                </tr>
              </thead>
              <tbody>
                {items.map((row) => (
                  <tr
                    key={row.request_id}
                    className={cn(
                      "border-b border-border/20 hover:bg-muted/40 transition-colors",
                      onSelectItem && "cursor-pointer"
                    )}
                    onClick={() => onSelectItem?.(row)}
                  >
                    <td className="py-2 px-2 text-muted-foreground whitespace-nowrap">
                      {new Date(row.at_ts * 1000).toLocaleTimeString()}
                    </td>
                    <td className="py-2 px-2 font-mono truncate max-w-[10ch]" title={row.model}>{row.model || "—"}</td>
                    <td className="py-2 px-2 truncate max-w-[14ch]" title={row.request_preview}>{row.request_preview || "—"}</td>
                    <td className="py-2 px-2 truncate max-w-[14ch]" title={row.response_preview}>{row.response_preview || "—"}</td>
                    <td className="py-2 px-2 text-right whitespace-nowrap">
                      In {formatNumber(row.input_tokens)} / Out {formatNumber(row.output_tokens)} · 합계 {formatNumber(row.total_tokens)}
                    </td>
                    <td className="py-2 px-2 text-right text-muted-foreground">{Math.round(row.duration_ms)}ms</td>
                    {onSelectItem && (
                      <td className="py-2 px-2 text-muted-foreground">
                        <ChevronRight className="w-4 h-4" />
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function ChatHistoryDetailDialog({
  item,
  open,
  onOpenChange,
  formatNumber,
}: {
  item: ChatHistoryItem | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  formatNumber: (n: number) => string;
}) {
  if (!item) return null;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-base sm:text-lg">챗 상세 · {new Date(item.at_ts * 1000).toLocaleString()}</DialogTitle>
          <DialogDescription>
            모델: {item.model || "—"} · 소요 {Math.round(item.duration_ms)}ms · 토큰 In {formatNumber(item.input_tokens)} / Out {formatNumber(item.output_tokens)} (합계 {formatNumber(item.total_tokens)})
          </DialogDescription>
        </DialogHeader>
        <div className="flex-1 overflow-y-auto space-y-4 text-sm">
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">Request ID</p>
            <p className="font-mono text-xs break-all text-foreground">{item.request_id}</p>
          </div>
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">요청 본문</p>
            <pre className="p-3 rounded-lg bg-muted/50 border border-border/40 overflow-x-auto overflow-y-auto max-h-40 whitespace-pre-wrap break-words text-foreground">
              {item.request_preview || "(없음)"}
            </pre>
          </div>
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">응답 본문</p>
            <pre className="p-3 rounded-lg bg-muted/50 border border-border/40 overflow-x-auto overflow-y-auto max-h-52 whitespace-pre-wrap break-words text-foreground">
              {item.response_preview || "(없음)"}
            </pre>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ===========================================
// Main Page
// ===========================================

const AUTO_REFRESH_MS = 10_000;

export default function MonitoringPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [detailItem, setDetailItem] = useState<ChatHistoryItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [dashboardRes, chatRes] = await Promise.all([
        fetch(`${API_URL}/api/monitoring/dashboard`),
        fetch(`${API_URL}/api/monitoring/chat-history?minutes=60&limit=50`),
      ]);

      if (!dashboardRes.ok) {
        throw new Error(`Dashboard HTTP ${dashboardRes.status}`);
      }

      const dashboardData = await dashboardRes.json();
      setData(dashboardData);
      setLastUpdate(new Date());
      setError(null);

      if (chatRes.ok) {
        const list = await chatRes.json();
        setChatHistory(Array.isArray(list) ? list : []);
      } else {
        setChatHistory([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchData, AUTO_REFRESH_MS);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchData]);

  return (
    <PageLayout
      title="Chat & AI Metrics"
      description="챗(AI 활용) 메트릭 전용 — 분산 시스템(Redis) 기반, /api/chat·/api/chat/* 구간만 집계. 요청·에러·응답시간·엔드포인트·토큰 세부 수치 표시"
    >
      {/* Header Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 sm:mb-6">
        <div className="flex items-center gap-2 text-xs sm:text-sm text-muted-foreground">
          {lastUpdate && (
            <span>Last updated: {lastUpdate.toLocaleTimeString()}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={cn(
              "text-xs sm:text-sm h-8 sm:h-9 border-border/40 bg-transparent hover:bg-muted/30",
              autoRefresh && "bg-muted/50 border-border/50"
            )}
          >
            <Activity className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
            Auto {autoRefresh ? "On" : "Off"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchData}
            disabled={loading}
            className="text-xs sm:text-sm h-8 sm:h-9 border-border/40 bg-transparent hover:bg-muted/30"
          >
            <RefreshCw
              className={cn(
                "w-3 h-3 sm:w-4 sm:h-4 mr-1",
                loading && "animate-spin"
              )}
            />
            Refresh
          </Button>
        </div>
      </div>

      {/* Error State — state.danger만 강조, 나머지 Minimal */}
      {error && (
        <div className="mb-4 bg-background border border-destructive/40 rounded-lg p-4 flex items-center gap-2 text-sm text-destructive">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading State — Minimal: bg.surface, border.subtle */}
      {loading && !data ? (
        <div className="grid gap-3 grid-cols-2 lg:grid-cols-5">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className={cn(CARD_CLASS, "p-4 animate-pulse")}
            >
              <div className="h-3 bg-muted rounded mb-2" />
              <div className="h-7 bg-muted rounded" />
            </div>
          ))}
        </div>
      ) : data ? (
        <div className="space-y-4 sm:space-y-6">
          {/* Summary Stats — 챗(AI 활용) 세부 수치 전부 (CACHE_AND_METRICS_POLICY §3.2) */}
          <div className="grid gap-3 sm:gap-4 grid-cols-2 lg:grid-cols-5">
            <StatCard
              title="Total Requests"
              value={formatNumber(data.summary.total_requests)}
              icon={Activity}
            />
            <StatCard
              title="Total Errors"
              value={formatNumber(data.summary.total_errors)}
              icon={AlertTriangle}
              className={data.summary.total_errors > 0 ? "border-destructive/40" : ""}
            />
            <StatCard
              title="Error Rate"
              value={data.summary.error_rate.toFixed(1)}
              unit="%"
              icon={AlertTriangle}
              className={data.summary.error_rate > 5 ? "border-destructive/40" : ""}
            />
            <StatCard
              title="Avg Response"
              value={data.summary.avg_response_time_ms.toFixed(0)}
              unit="ms"
              icon={Clock}
            />
            <StatCard
              title="P95 Response"
              value={data.summary.p95_response_time_ms.toFixed(0)}
              unit="ms"
              icon={Zap}
            />
          </div>

          {/* Charts — Technical Elegance: 다양한 그래프, 데이터 중심·깊이감 유지 */}
          <div className="grid gap-4 sm:gap-6 lg:grid-cols-2">
            <RequestTrendChart data={data.request_trend} />
            <HealthIndicator health={data.health} />
          </div>

          <div className="grid gap-4 sm:gap-6 lg:grid-cols-2">
            <TokenUsagePieChart usage={data.token_usage} />
            <ResponseTimeBarChart summary={data.summary} />
          </div>

          <div className="grid gap-4 sm:gap-6 lg:grid-cols-2">
            <EndpointTable endpoints={data.top_endpoints} />
            <TokenUsageTable usage={data.token_usage} />
          </div>

          {/* 챗 이력 · 시간대별 Bar + 상세 테이블 */}
          <div className="grid gap-4 sm:gap-6 lg:grid-cols-2">
            <ChatHistoryTimeChart items={chatHistory} />
            <ChatHistoryTable items={chatHistory} onSelectItem={setDetailItem} />
          </div>
          <ChatHistoryDetailDialog
            item={detailItem}
            open={!!detailItem}
            onOpenChange={(o) => !o && setDetailItem(null)}
            formatNumber={formatNumber}
          />
        </div>
      ) : (
        <div className="text-center py-12 text-muted-foreground">
          No monitoring data available
        </div>
      )}
    </PageLayout>
  );
}

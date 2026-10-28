"use client";

import { Activity, AlertTriangle, Camera, Clock, type LucideIcon } from "lucide-react";
import { formatDistance } from "date-fns";
import { zhTW } from "date-fns/locale";
import type { DetectionStats } from "@/lib/api";

/** Parse a naive datetime string as Taiwan time (GMT+8) and return a UTC Date. */
function parseTW(naive: string): Date {
  return new Date(naive + "+08:00");
}

interface StatsCardsProps {
  stats: DetectionStats | undefined;
  isLoading: boolean;
}

interface StatCard {
  title: string;
  value: React.ReactNode;
  icon: LucideIcon;
  sub: string;
  color: string;
}

export function StatsCards({ stats, isLoading }: StatsCardsProps) {
  const cards: StatCard[] = [
    {
      title: "今日偵測",
      value: stats?.total_today ?? "--",
      icon: AlertTriangle,
      sub: "過去 24 小時",
      color: "text-orange-500",
    },
    {
      title: "累計偵測",
      value: stats?.total_all ?? "--",
      icon: Activity,
      sub: "所有時間",
      color: "text-blue-500",
    },
    {
      title: "最近發現",
      value: stats?.last_detection
        ? formatDistance(parseTW(stats.last_detection), new Date(), {
            addSuffix: true,
            locale: zhTW,
          })
        : "無",
      icon: Clock,
      sub: "距上次偵測",
      color: "text-green-500",
    },
    {
      title: "運作中攝影機",
      value: stats?.cameras_active ?? 0,
      icon: Camera,
      sub: "RTSP 串流",
      color: "text-violet-500",
    },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((card) => (
        <StatTile
          key={card.title}
          {...card}
          value={isLoading ? (
            <span className="inline-block h-7 w-16 animate-pulse rounded bg-muted" />
          ) : card.value}
        />
      ))}
    </div>
  );
}

export function StatTile({
  title,
  value,
  sub,
  icon: Icon,
  color,
}: StatCard) {
  return (
    <div className="rounded-lg border bg-white px-5 py-4">
      <div className="flex items-center justify-between">
        <p className="text-[13px] text-muted-foreground">{title}</p>
        <Icon className={`h-4 w-4 ${color}`} />
      </div>
      <div className="mt-2 text-2xl font-semibold">{value}</div>
      <p className="mt-1 text-xs text-muted-foreground">{sub}</p>
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";
import { TriangleAlertIcon } from "lucide-react";

import { formatBytes } from "../lib/format";
import { rpcClient } from "../lib/rpc";
import { cn } from "../lib/utils";
import { usedRatio, type ResourceUsage } from "../../shared/hardware";
import { useRouter } from "../stores/router";
import { useServedStore } from "../stores/served";
import { useServerStore } from "../stores/server";
import { useT } from "../stores/ui-lang";

/**
 * 媒体服务告警：端口被另一个数据目录的实例占着时，媒体预览会失败或串到对方数据上，
 * 界面必须说出来——不然用户只会看到一个"文件不存在"的播放器 / 裂图。
 * 服务被挡住时主进程会每几秒重试，对方退出后自动接管，这里的状态随之消失。
 */
function MediaStatusChip() {
  const t = useT();
  const mediaStatus = useServerStore((s) => s.mediaStatus);
  if (mediaStatus.state !== "blocked") return null;
  return (
    <span
      title={t("media.status.blockedHint")}
      className="flex items-center gap-1 rounded-full border border-destructive/40 bg-destructive/10 px-2 py-0.5 text-[10px] font-medium text-destructive"
    >
      <TriangleAlertIcon className="size-3" />
      {t("media.status.blocked")}
    </span>
  );
}

// ---------------------------------------------------------------------------
// 内存 / 显存两段
// ---------------------------------------------------------------------------

/** 剩余比例的电量条颜色：低于 10% 报警（destructive），10–30% 偏紧（amber），其余正常。 */
function freeBarClass(ratio: number | null): string {
  if (ratio === null) return "bg-muted";
  if (ratio < 0.1) return "bg-destructive";
  if (ratio < 0.3) return "bg-amber-500";
  return "bg-primary";
}

/** tooltip 的一段：总量 / 已用 / 剩余，缺哪个读数说哪个（读不出就是「—」，不猜数）。 */
function usageTitleLine(label: string, usage: ResourceUsage, t: (key: string, vars?: Record<string, string>) => string): string {
  const total = usage.totalBytes ?? 0;
  const free = usage.freeBytes ?? 0;
  const used = total > 0 ? Math.max(0, total - free) : 0;
  return t(label, {
    total: formatBytes(total),
    used: formatBytes(used),
    free: formatBytes(free),
  });
}

/**
 * 内存 / 显存一段：标签 + 细电量条（填充 = 剩余比例）+ 剩余百分比 + 剩余量。
 *
 * 容器查询的容器是**胶囊按钮本身**（本段只是 `@[...]` 消费者，自己不能再是
 * `@container`——inline-size 容器宽度会塌成 0，条件永远不成立）。
 * 档位随按钮宽度从大到小依次隐藏：剩余量（<230px）→ 电量条（<170px）→
 * 整段含百分比（<130px），最窄时只剩「运行模型数:n」，胶囊永远不会空。
 */
/**
 * 按资源段数选的容器档位（按钮宽度）。Tailwind 只认字面量 class，所以两套档位写死在这里：
 * 一段（Apple 统一内存）内容约 230px；两段（独显）约 370px，阈值随之放大，
 * 否则两段时剩余量永远挤不下、一段时按钮又空出一截。
 */
const SEGMENT_TIERS = {
  one: {
    button: "flex-[0_1_236px]",
    segment: "@[130px]:flex",
    pct: "@[130px]:inline",
    bar: "@[170px]:inline-block",
    left: "@[230px]:inline",
  },
  two: {
    button: "flex-[0_1_380px]",
    segment: "@[190px]:flex",
    pct: "@[190px]:inline",
    bar: "@[270px]:inline-block",
    left: "@[370px]:inline",
  },
} as const;

type SegmentTier = (typeof SEGMENT_TIERS)[keyof typeof SEGMENT_TIERS];

function ResourceSegment({
  label,
  usage,
  title,
  tier,
}: {
  label: string;
  usage: ResourceUsage;
  title: string;
  tier: SegmentTier;
}) {
  const used = usedRatio(usage);
  // 电量条与百分比都表达「剩余」：剩余比例 = 1 - 已用比例。
  const freeRatio = used === null ? null : 1 - used;
  const pct = freeRatio === null ? null : Math.round(freeRatio * 100);
  const free = usage.freeBytes;
  return (
    <span className={cn("hidden flex-none items-center gap-1", tier.segment)} title={title}>
      <span className="flex-none text-muted-foreground">{label}</span>
      {/* 电量条（填充 = 剩余比例）：按钮不足 170px 时隐藏（先于整段消失）。 */}
      <span
        className={cn("hidden h-1 w-8 overflow-hidden rounded-full bg-muted", tier.bar)}
        aria-hidden
      >
        <span
          className={cn("block h-full", freeBarClass(freeRatio))}
          style={{ width: `${freeRatio === null ? 0 : Math.round(freeRatio * 100)}%` }}
        />
      </span>
      <span
        className={cn(
          "hidden flex-none tabular-nums",
          tier.pct,
          freeRatio === null
            ? "text-muted-foreground"
            : freeRatio < 0.1
              ? "text-destructive"
              : freeRatio < 0.3
                ? "text-amber-500"
                : "text-foreground",
        )}
      >
        {pct === null ? "—" : `${pct}%`}
      </span>
      {/* 剩余量：按钮不足 230px 时隐藏（最先消失的次要信息）。 */}
      <span className={cn("hidden flex-none text-muted-foreground", tier.left)}>
        {free !== null ? formatBytes(free) : ""}
      </span>
    </span>
  );
}

/**
 * 顶栏状态胶囊：本地模式显示**运行中的模型数**（多开时不再是一个笼统的「运行中」）
 * 加内存 / 显存余量（Apple 统一内存机器只有内存一段），点一下进控制台 —— 启停 / 卸载都在那儿。
 * 云端模式显示连通性。
 */
export function StatusPill() {
  const t = useT();
  const serverStatus = useServerStore((s) => s.status);
  const servedModels = useServedStore((s) => s.models);
  const setRoute = useRouter((s) => s.setRoute);

  const { data, isLoading } = useQuery({
    queryKey: ["connection-status"],
    queryFn: () => rpcClient.checkConnection(undefined),
    refetchInterval: 30_000,
  });

  const { data: settingsData } = useQuery({
    queryKey: ["settings"],
    queryFn: () => rpcClient.getSettings(undefined),
  });

  const isLocal = (settingsData?.settings?.SERVER_MODE ?? "local") === "local";

  // 资源余量只有本地模式才有意义（云端模型不吃本机内存），所以只在本地模式轮询。
  const { data: usage } = useQuery({
    queryKey: ["resource-usage"],
    queryFn: () => rpcClient.getResourceUsage(undefined),
    enabled: isLocal,
    refetchInterval: 5_000,
  });

  const openConsole = () => setRoute({ path: "settings", tab: "logs" });

  if (isLocal) {
    const running = servedModels.filter((m) => m.status === "running").length;
    const loading = servedModels.filter(
      (m) => m.status === "starting" || m.status === "downloading",
    ).length;
    const failed = servedModels.filter((m) => m.status === "error").length;

    // 有模型在跑显示「运行模型数:n」，否则保留原来的启动中 / 出错 / 已停止文字；余量段两种情况都显示。
    const label =
      running > 0
        ? t("server.status.runningModels", { n: String(running) })
        : loading > 0
          ? t("server.startingModel")
          : failed > 0
            ? t("server.status.error")
            : t(`server.status.${serverStatus === "running" ? "stopped" : serverStatus}`);

    const dotClass =
      running > 0
        ? "bg-green-500"
        : loading > 0 || serverStatus === "starting" || serverStatus === "downloading"
          ? "animate-pulse bg-amber-500"
          : failed > 0 || serverStatus === "error"
            ? "bg-destructive"
            : "bg-muted-foreground";

    // 统一内存机器（Apple Silicon / APU）只有内存一段；NVIDIA 独显两段。
    const segments: {
      label: string;
      usage: ResourceUsage;
      title: string;
    }[] = [];
    if (usage) {
      segments.push({
        label: t("server.status.ram"),
        usage: usage.ram,
        title: usageTitleLine("server.status.ramTitle", usage.ram, t) +
          (usage.unifiedMemory ? t("server.status.unifiedHint") : ""),
      });
      if (usage.vram !== null) {
        segments.push({
          label: t("server.status.vram"),
          usage: usage.vram,
          title: usageTitleLine("server.status.vramTitle", usage.vram, t),
        });
      }
    }

    const tier = segments.length === 0 ? null : segments.length > 1 ? SEGMENT_TIERS.two : SEGMENT_TIERS.one;

    return (
      <>
        <MediaStatusChip />
        <button
          type="button"
          onClick={openConsole}
          title={t("console.open")}
          className={cn(
            "flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors hover:bg-muted",
            // 有余量段时按钮做容器：container-type 让宽度不再由内容撑开，所以给一个可收缩的基准宽，
            // 顶栏挤的时候按钮变窄，段内按档位依次藏掉剩余量 → 进度条 → 百分比 / 整段。
            tier && ["@container min-w-[110px] overflow-hidden", tier.button],
          )}
        >
          <div className={cn("size-1.5 flex-none rounded-full", dotClass)} />
          <span className="flex-none">{label}</span>
          {tier && (
            <span className="flex min-w-0 items-center gap-1.5">
              {segments.map((seg) => (
                <ResourceSegment key={seg.label} label={seg.label} usage={seg.usage} title={seg.title} tier={tier} />
              ))}
            </span>
          )}
        </button>
      </>
    );
  }

  const connected = data?.connected ?? false;
  const label = isLoading ? "Checking…" : connected ? "Connected" : "Disconnected";

  return (
    <>
      <MediaStatusChip />
      <button
        type="button"
        onClick={openConsole}
        title={t("console.open")}
        className="flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors hover:bg-muted"
      >
        <div
          className={cn(
            "size-1.5 rounded-full",
            isLoading
              ? "animate-pulse bg-muted-foreground"
              : connected
                ? "bg-green-500"
                : "bg-destructive",
          )}
        />
        {label}
      </button>
    </>
  );
}

/**
 * 下载源路由的运行期实现（契约见 shared/net-sources.ts）。
 *
 * 为什么要探测而不是看地区：国内用户很多开着系统代理（bun/proxy.ts 把全局 fetch 包了一层），
 * 这时直连 huggingface.co / github.com 反而比镜像快；没开代理的则官方源要么连不上、要么
 * 慢到不可用。所以这里并行探一遍官方源与各镜像（经同一个带代理的 fetch，测到的就是真实
 * 下载会走的路），按「连得上 + 首包延迟」给出 SourcePlan：
 *
 *   - auto：官方 HF 或 GitHub 连不上 → 国内加速；地区信号指向大陆且镜像明显更快 → 国内加速；
 *     否则官方直连。一个都探不通（离线 / 探测本身失败）就按地区信号猜。
 *   - cn / global（用户强制）：模式不再探测决定，但仍探测，用来给同一模式内的候选排序。
 *
 * 结论缓存 10 分钟，并发调用共用同一次探测；下载失败时调用方 reportSourceFailure(url)，
 * 那个主机会被降级、下次取计划时重新探测。
 *
 * 旧的 MLX_HF_ENDPOINT 设置仍然有效（MLX 引擎自己读），但新代码一律走这里的 sourceEnv。
 */

import { logEvent } from "./app-log";
import { getSetting } from "./db/settings";
import type { DownloadRegionSetting, SourceKind, SourcePlan, SourceProbe } from "../shared/net-sources";
import { DOWNLOAD_REGION_VALUES } from "../shared/net-sources";

// ---------------------------------------------------------------------------
// 源清单
// ---------------------------------------------------------------------------

export const HF_OFFICIAL = "https://huggingface.co";
export const HF_MIRRORS = ["https://hf-mirror.com"] as const;

export const PYPI_OFFICIAL = "https://pypi.org/simple";
export const PYPI_MIRRORS = [
  "https://mirrors.aliyun.com/pypi/simple",
  "https://pypi.tuna.tsinghua.edu.cn/simple",
] as const;

/** 直连 GitHub 在 githubPrefixes 里记作空串；探测明细里记作这个地址，界面才有东西可显示。 */
export const GITHUB_OFFICIAL = "https://github.com";

/**
 * GitHub 加速镜像（前缀式：`<mirror><原始 URL>`）。按实测可用性排的初始顺序，
 * 实际顺序以探测为准。mirror-download.ts 的 GITHUB_MIRRORS 就是这份。
 */
export const GITHUB_PREFIX_MIRRORS = [
  "https://gh-proxy.com/",
  "https://ghfast.top/",
  "https://ghproxy.net/",
] as const;

export const MODELSCOPE_URL = "https://modelscope.cn";

const USTC_BREW = "https://mirrors.ustc.edu.cn/homebrew-bottles";

/** uv 下载独立 Python 的原始地址（uv 会在后面拼 `/<tag>/<asset>`）。 */
const PYTHON_BUILD_STANDALONE = "https://github.com/astral-sh/python-build-standalone/releases/download";

/**
 * 经前缀镜像探测用的小文件：raw.githubusercontent 上一个长期存在的几 KB 文本。
 * 镜像「有响应」不代表它能代理 GitHub（挂了的镜像常回 200 的 HTML 首页），所以前缀镜像
 * 必须真的把这个文件取回来（2xx 且不是网页）才算通。
 */
const GITHUB_PROBE_FILE = "https://raw.githubusercontent.com/github/gitignore/main/README.md";

type ProbeTarget = {
  kind: SourceKind;
  /** 源本身（进 SourcePlan 的值）。 */
  url: string;
  official: boolean;
  /** 实际请求的地址。 */
  probeUrl: string;
  /** true = 必须 2xx 且不是 HTML；false = 只要有 HTTP 响应（<500）就算网络通。 */
  strict: boolean;
};

function probeTargets(): ProbeTarget[] {
  const t: ProbeTarget[] = [
    { kind: "hf", url: HF_OFFICIAL, official: true, probeUrl: `${HF_OFFICIAL}/`, strict: false },
    ...HF_MIRRORS.map((u) => ({ kind: "hf" as const, url: u, official: false, probeUrl: `${u}/`, strict: false })),
    { kind: "modelscope", url: MODELSCOPE_URL, official: false, probeUrl: `${MODELSCOPE_URL}/`, strict: false },
    // PyPI 用 strict：镜像同步坏掉时会对 /simple/pip/ 回 404，那种镜像不能排前面。
    { kind: "pypi", url: PYPI_OFFICIAL, official: true, probeUrl: `${PYPI_OFFICIAL}/pip/`, strict: true },
    ...PYPI_MIRRORS.map((u) => ({ kind: "pypi" as const, url: u, official: false, probeUrl: `${u}/pip/`, strict: true })),
    { kind: "github", url: GITHUB_OFFICIAL, official: true, probeUrl: `${GITHUB_OFFICIAL}/robots.txt`, strict: false },
    ...GITHUB_PREFIX_MIRRORS.map((p) => ({
      kind: "github" as const,
      url: p,
      official: false,
      probeUrl: `${p}${GITHUB_PROBE_FILE}`,
      strict: true,
    })),
    { kind: "homebrew", url: USTC_BREW, official: false, probeUrl: `${USTC_BREW}/api/formula.jws.json`, strict: false },
  ];
  return t;
}

// ---------------------------------------------------------------------------
// 探测
// ---------------------------------------------------------------------------

const PROBE_TIMEOUT_MS = 3_000;
const PLAN_TTL_MS = 10 * 60_000;
/** reportSourceFailure 之后这个主机被降级多久（与计划缓存同长，下一轮探测前一直有效）。 */
const FAILURE_TTL_MS = 10 * 60_000;

export type ProbeResult = { ok: boolean; latencyMs: number | null };
/** 单个地址的探测函数；测试里整体替换掉，不碰网络。 */
export type ProbeFn = (url: string, strict: boolean) => Promise<ProbeResult>;

/**
 * 真实探测：走全局 fetch（已被 bun/proxy.ts 包上系统 / 自定义代理），首包即计时。
 * 用 GET + Range 而不是 HEAD：部分镜像对 HEAD 回 405 / 不转发，GET 取几十字节足够且判定一致。
 */
export const defaultProbe: ProbeFn = async (url, strict) => {
  const started = performance.now();
  try {
    const res = await fetch(url, {
      method: "GET",
      redirect: strict ? "follow" : "manual",
      // identity：只取前 256 字节就掐断，压缩流被截断时 Bun 会往 stderr 打 ZlibError。
      headers: { Range: "bytes=0-255", "Accept-Encoding": "identity" },
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    const latencyMs = Math.round(performance.now() - started);
    await res.body?.cancel().catch(() => {});
    if (strict) {
      const type = res.headers.get("content-type") ?? "";
      const ok = res.status >= 200 && res.status < 300 && (url.includes("/simple/") || !type.includes("text/html"));
      return { ok, latencyMs: ok ? latencyMs : null };
    }
    return res.status < 500 ? { ok: true, latencyMs } : { ok: false, latencyMs: null };
  } catch {
    return { ok: false, latencyMs: null };
  }
};

// ---------------------------------------------------------------------------
// 地区信号
// ---------------------------------------------------------------------------

const CN_TIMEZONES = new Set(["Asia/Shanghai", "Asia/Chongqing", "Asia/Chungking", "Asia/Harbin", "Asia/Urumqi", "Asia/Kashgar", "PRC"]);

/** zh_CN / zh-CN / zh-Hans-CN / zh_Hans_CN（带编码后缀也算）；zh_TW / zh_HK / zh-Hant 不算。 */
function isCnLocaleTag(tag: string | undefined): boolean {
  if (!tag) return false;
  return /^zh[-_](hans[-_])?cn\b/i.test(tag.trim());
}

/**
 * 地区信号是否指向中国大陆。纯函数：输入由 cnLocaleSignals() 收集，测试直接喂。
 * 只作「探测失败时的猜测」与「镜像明显更快时的偏好」，从不单独决定走镜像。
 */
export function detectCnLocale(signals: { timeZone?: string; langs?: (string | undefined)[] }): boolean {
  if (signals.timeZone && CN_TIMEZONES.has(signals.timeZone)) return true;
  return (signals.langs ?? []).some(isCnLocaleTag);
}

let cachedAppleLocale: string | undefined | null = null;
function appleLocale(): string | undefined {
  if (cachedAppleLocale !== null) return cachedAppleLocale;
  cachedAppleLocale = undefined;
  if (process.platform !== "darwin") return undefined;
  try {
    // GUI 启动的 app 通常没有 LANG，系统语言只在这里（一次十几毫秒，结果缓存）。
    const r = Bun.spawnSync(["defaults", "read", "-g", "AppleLocale"], { stdout: "pipe", stderr: "ignore" });
    const v = r.stdout.toString().trim();
    if (r.exitCode === 0 && v) cachedAppleLocale = v;
  } catch {
    // defaults 不可用就当没有这个信号
  }
  return cachedAppleLocale;
}

function cnLocaleSignals(): { timeZone?: string; langs: (string | undefined)[] } {
  let timeZone: string | undefined;
  let intlLocale: string | undefined;
  try {
    const o = Intl.DateTimeFormat().resolvedOptions();
    timeZone = o.timeZone;
    intlLocale = o.locale;
  } catch {
    // ignore
  }
  const e = process.env;
  return { timeZone, langs: [e.LC_ALL, e.LC_MESSAGES, e.LANG, appleLocale(), intlLocale] };
}

// ---------------------------------------------------------------------------
// 决策（纯函数）
// ---------------------------------------------------------------------------

export type PlanInput = {
  region: DownloadRegionSetting;
  cnLocale: boolean;
  /** 探测明细；空数组 = 还没探（peek 的猜测计划）。 */
  probes: SourceProbe[];
  overrides?: { hf?: string; pypi?: string; github?: string };
  /** 最近报告失败的主机（降级到本类末尾，但官方兜底规则仍然成立）。 */
  failedHosts?: ReadonlySet<string>;
  now?: number;
};

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

const trimSlash = (u: string) => u.trim().replace(/\/+$/, "");
const withSlash = (u: string) => (u.trim().endsWith("/") ? u.trim() : `${u.trim()}/`);

/** 镜像明显更快：官方首包 > 2× 最快镜像 且 > 800ms（光是「慢一点」不值得换源）。 */
function mirrorsClearlyFaster(official: SourceProbe | undefined, mirrors: SourceProbe[]): boolean {
  const best = mirrors.filter((p) => p.ok && p.latencyMs != null).sort((a, b) => a.latencyMs! - b.latencyMs!)[0];
  if (!official?.ok || official.latencyMs == null || !best) return false;
  return official.latencyMs > 2 * best.latencyMs! && official.latencyMs > 800;
}

/**
 * 同类候选排序。
 * cn：镜像按「未报失败 → 可达 → 延迟」排，官方永远在最后兜底。
 * global：官方可达且没报失败就排第一；否则镜像在前、官方最后。
 * 没探测过（probe 缺失）的项视为「未知」，排在可达之后、不可达之前，保持原始顺序。
 */
function orderKind(
  official: string,
  mirrors: readonly string[],
  mode: "cn" | "global",
  probeOf: (url: string) => SourceProbe | undefined,
  failed: ReadonlySet<string>,
): string[] {
  const rank = (u: string): [number, number, number] => {
    const p = probeOf(u);
    const bad = failed.has(hostOf(u) || u) ? 1 : 0;
    const reach = !p ? 1 : p.ok ? 0 : 2;
    return [bad, reach, p?.latencyMs ?? Number.MAX_SAFE_INTEGER];
  };
  const sorted = mirrors
    .map((u, i) => ({ u, i, r: rank(u) }))
    .sort((a, b) => a.r[0] - b.r[0] || a.r[1] - b.r[1] || a.r[2] - b.r[2] || a.i - b.i)
    .map((x) => x.u);
  if (mode === "global") {
    const r = rank(official);
    if (r[0] === 0 && r[1] <= 1) return [official, ...sorted];
  }
  return [...sorted, official];
}

/** 覆盖值放最前（去重）；空串不覆盖。 */
function withOverride(list: string[], override: string | undefined): string[] {
  if (override === undefined || override === null) return list;
  return [override, ...list.filter((u) => u !== override)];
}

/** 由探测结果与设置算出 SourcePlan。纯函数，决策逻辑的测试都打在这里。 */
export function decidePlan(input: PlanInput): SourcePlan {
  const { region, cnLocale, probes } = input;
  const failed = input.failedHosts ?? new Set<string>();
  const probeOf = (url: string) => probes.find((p) => p.url === url);
  const officialOf = (kind: SourceKind) => probes.find((p) => p.kind === kind && p.official);
  const mirrorsOf = (kind: SourceKind) => probes.filter((p) => p.kind === kind && !p.official);

  let mode: "cn" | "global";
  let decidedBy: SourcePlan["decidedBy"];
  if (region === "cn" || region === "global") {
    mode = region;
    decidedBy = "setting";
  } else if (!probes.some((p) => p.ok)) {
    // 一个都不通：要么离线，要么还没探。探测没有信息量，只能按地区猜。
    mode = cnLocale ? "cn" : "global";
    decidedBy = "locale-guess";
  } else {
    const hf = officialOf("hf");
    const gh = officialOf("github");
    const officialDown = (hf && !hf.ok) || (gh && !gh.ok);
    const faster =
      cnLocale &&
      (mirrorsClearlyFaster(hf, mirrorsOf("hf")) || mirrorsClearlyFaster(gh, mirrorsOf("github")));
    mode = officialDown || faster ? "cn" : "global";
    decidedBy = "probe";
  }

  const ov = input.overrides ?? {};
  const hfEndpoints = withOverride(
    orderKind(HF_OFFICIAL, HF_MIRRORS, mode, probeOf, failed),
    ov.hf ? trimSlash(ov.hf) : undefined,
  );
  const pypiIndexes = withOverride(
    orderKind(PYPI_OFFICIAL, PYPI_MIRRORS, mode, probeOf, failed),
    ov.pypi ? trimSlash(ov.pypi) : undefined,
  );
  // GitHub 的官方项在候选里是空串（= 不加前缀直连），探测明细里是 https://github.com。
  const ghProbeOf = (u: string) => probeOf(u === "" ? GITHUB_OFFICIAL : u);
  const ghFailed = new Set(failed);
  if (failed.has(hostOf(GITHUB_OFFICIAL))) ghFailed.add("");
  const githubPrefixes = withOverride(
    orderKind("", GITHUB_PREFIX_MIRRORS, mode, ghProbeOf, ghFailed),
    ov.github ? withSlash(ov.github) : undefined,
  );

  const homebrewEnv: Record<string, string> = {};
  if (mode === "cn") {
    const brew = probeOf(USTC_BREW);
    // 没探过（猜测计划）也给：国内不设这组变量 brew 基本装不动，设了最坏也只是和不设一样慢。
    if (!brew || brew.ok) {
      homebrewEnv.HOMEBREW_API_DOMAIN = `${USTC_BREW}/api`;
      homebrewEnv.HOMEBREW_BOTTLE_DOMAIN = USTC_BREW;
    }
    const pip = pypiIndexes.find((u) => u !== PYPI_OFFICIAL);
    if (pip) homebrewEnv.HOMEBREW_PIP_INDEX_URL = pip;
  }

  return {
    mode,
    decidedBy,
    cnLocale,
    modelSource: mode === "cn" ? "modelscope" : "huggingface",
    hfEndpoints,
    pypiIndexes,
    githubPrefixes,
    homebrewEnv,
    probes,
    at: input.now ?? Date.now(),
  };
}

// ---------------------------------------------------------------------------
// 子进程环境 / GitHub 候选
// ---------------------------------------------------------------------------

/**
 * 给子进程（pip / uv / huggingface_hub / llama-server -hf / brew）的环境变量。
 *
 * - HF_ENDPOINT：huggingface_hub、MLX、vLLM 等 Python 侧都认它。
 * - MODEL_ENDPOINT：llama.cpp（libllama-common）先读 MODEL_ENDPOINT、再读 HF_ENDPOINT，
 *   默认值是带尾斜杠的 `https://huggingface.co/`，它直接在后面拼路径 —— 所以这里带尾斜杠给。
 *   （已对本机引擎的 libllama-common.dylib 跑 strings 确认两个变量名都在。）
 * - PIP_INDEX_URL / UV_DEFAULT_INDEX：pip 与 uv 的默认索引。
 * - UV_PYTHON_INSTALL_MIRROR：uv 装独立 Python 时从 GitHub release 拉，国内要走前缀镜像。
 * 官方值一律不设：不设 = 工具自己的默认，免得把用户自己的配置（pip.conf 等）盖掉。
 */
export function sourceEnv(plan: SourcePlan): Record<string, string> {
  const env: Record<string, string> = {};
  const hf = plan.hfEndpoints[0];
  if (hf && trimSlash(hf) !== HF_OFFICIAL) {
    env.HF_ENDPOINT = trimSlash(hf);
    env.MODEL_ENDPOINT = `${trimSlash(hf)}/`;
  }
  const pypi = plan.pypiIndexes[0];
  if (pypi && trimSlash(pypi) !== PYPI_OFFICIAL) {
    env.PIP_INDEX_URL = pypi;
    env.UV_DEFAULT_INDEX = pypi;
  }
  if (plan.mode === "cn") {
    const prefix = plan.githubPrefixes.find((p) => p !== "") ?? GITHUB_PREFIX_MIRRORS[0];
    env.UV_PYTHON_INSTALL_MIRROR = `${prefix}${PYTHON_BUILD_STANDALONE}`;
  }
  Object.assign(env, plan.homebrewEnv);
  return env;
}

/** GitHub 资源的候选 URL（按计划的前缀顺序拼好；空串前缀 = 原始 URL 直连）。 */
export function githubCandidates(url: string, plan?: SourcePlan): string[] {
  const p = plan ?? peekSourcePlan();
  const out: string[] = [];
  for (const prefix of p.githubPrefixes) {
    const u = `${prefix}${url}`;
    if (!out.includes(u)) out.push(u);
  }
  if (!out.includes(url)) out.push(url); // 直连永远兜底，哪怕覆盖设置把它挤掉了
  return out;
}

// ---------------------------------------------------------------------------
// 缓存 / 并发 / 失败上报
// ---------------------------------------------------------------------------

type Deps = {
  probe: ProbeFn;
  settings: () => { region: DownloadRegionSetting; hf: string; pypi: string; github: string };
  cnLocale: () => boolean;
  now: () => number;
};

function readSettings(): ReturnType<Deps["settings"]> {
  const raw = getSetting("DOWNLOAD_REGION");
  const region = (DOWNLOAD_REGION_VALUES as readonly string[]).includes(raw) ? (raw as DownloadRegionSetting) : "auto";
  return {
    region,
    hf: getSetting("DOWNLOAD_HF_ENDPOINT").trim(),
    pypi: getSetting("DOWNLOAD_PYPI_INDEX").trim(),
    github: getSetting("DOWNLOAD_GITHUB_MIRROR").trim(),
  };
}

/**
 * 测试进程（test-preload 设 NODE_ENV=test）里默认不联网、不看本机地区：别的模块的测试
 * 顺手调到 peekSourcePlan / getSourcePlan 时，不该在后台打真实探测（慢、结果随网络变），
 * 也不该因为跑测试的机器在国内就拿到不同的计划。统一落到「官方直连 · 按地区猜」，
 * 需要别的计划的测试用 __setNetSourcesDepsForTest / __setSourcePlanForTest 注入。
 */
const IS_TEST = process.env.NODE_ENV === "test";

const defaultDeps: Deps = {
  probe: IS_TEST ? async () => ({ ok: false, latencyMs: null }) : defaultProbe,
  settings: readSettings,
  cnLocale: IS_TEST ? () => false : () => detectCnLocale(cnLocaleSignals()),
  now: () => Date.now(),
};
let deps: Deps = defaultDeps;

let cached: { plan: SourcePlan; sig: string; stale: boolean } | null = null;
let inFlight: { sig: string; promise: Promise<SourcePlan> } | null = null;
/** 探测世代：失败上报 / 测试重置后，正在跑的旧探测结果不再写回缓存。 */
let generation = 0;
const failedHosts = new Map<string, number>();

function activeFailures(now: number): Set<string> {
  const out = new Set<string>();
  for (const [host, at] of failedHosts) {
    if (now - at < FAILURE_TTL_MS) out.add(host);
    else failedHosts.delete(host);
  }
  return out;
}

/** 设置指纹：用户改了下载源 / 覆盖地址，缓存立即作废（不必等 10 分钟）。 */
function settingsSig(s: ReturnType<Deps["settings"]>): string {
  return `${s.region}|${s.hf}|${s.pypi}|${s.github}`;
}

function planFrom(s: ReturnType<Deps["settings"]>, probes: SourceProbe[]): SourcePlan {
  const now = deps.now();
  return decidePlan({
    region: s.region,
    cnLocale: deps.cnLocale(),
    probes,
    overrides: { hf: s.hf || undefined, pypi: s.pypi || undefined, github: s.github || undefined },
    failedHosts: activeFailures(now),
    now,
  });
}

async function runProbe(s: ReturnType<Deps["settings"]>, sig: string, gen: number): Promise<SourcePlan> {
  const targets = probeTargets();
  const probes: SourceProbe[] = await Promise.all(
    targets.map(async (t) => {
      const r = await deps.probe(t.probeUrl, t.strict).catch((): ProbeResult => ({ ok: false, latencyMs: null }));
      return { url: t.url, ok: r.ok, latencyMs: r.ok ? r.latencyMs : null, kind: t.kind, official: t.official };
    }),
  );
  const plan = planFrom(s, probes);
  if (gen === generation) cached = { plan, sig, stale: false };
  logEvent({
    level: "info",
    source: "download",
    event: "net-sources.plan",
    message: `下载源：${plan.mode === "cn" ? "国内加速" : "官方直连"}（${plan.decidedBy}）· HF ${hostOf(plan.hfEndpoints[0] ?? "")} · PyPI ${hostOf(plan.pypiIndexes[0] ?? "")} · GitHub ${plan.githubPrefixes[0] ? hostOf(plan.githubPrefixes[0]) : "直连"}`,
    detail: {
      mode: plan.mode,
      decidedBy: plan.decidedBy,
      cnLocale: plan.cnLocale,
      region: s.region,
      hfEndpoints: plan.hfEndpoints,
      pypiIndexes: plan.pypiIndexes,
      githubPrefixes: plan.githubPrefixes,
      probes: probes.map((p) => ({ url: p.url, ok: p.ok, ms: p.latencyMs })),
      failedHosts: [...activeFailures(deps.now())],
    },
  });
  return plan;
}

/**
 * 取当前下载源计划：缓存 10 分钟；并发调用共用一次探测；refresh = 忽略缓存重探。
 * 探测最坏耗时 ≈ 单次超时（3s，全部并行），调用方不必另加超时。
 */
export async function getSourcePlan(opts?: { refresh?: boolean }): Promise<SourcePlan> {
  const s = deps.settings();
  const sig = settingsSig(s);
  const now = deps.now();
  if (!opts?.refresh && cached && cached.sig === sig && !cached.stale && now - cached.plan.at < PLAN_TTL_MS) {
    return cached.plan;
  }
  if (inFlight && inFlight.sig === sig) return inFlight.promise;
  const gen = generation;
  const promise = runProbe(s, sig, gen).finally(() => {
    if (inFlight?.promise === promise) inFlight = null;
  });
  inFlight = { sig, promise };
  return promise;
}

/**
 * 同步取计划（给不方便 await 的地方，如拼 spawn 环境）：有缓存用缓存（过期也先用，
 * 同时后台刷新）；没有就按地区信号给一份不联网的猜测计划，并在后台发起探测。
 */
export function peekSourcePlan(): SourcePlan {
  const s = deps.settings();
  const sig = settingsSig(s);
  const fresh = cached && cached.sig === sig && !cached.stale && deps.now() - cached.plan.at < PLAN_TTL_MS;
  if (!fresh) void getSourcePlan().catch(() => {});
  if (cached && cached.sig === sig) return cached.plan;
  // 设置变了但探测结果还在：沿用旧探测明细按新设置重算，比纯猜准。
  return planFrom(s, cached?.plan.probes ?? []);
}

/**
 * 下载方报告某个源失败（整条 URL 即可，取主机名）：该主机降级 10 分钟，
 * 缓存标记为过期 —— peek 仍返回旧计划（别退回纯猜），下一次 getSourcePlan 重新探测。
 */
export function reportSourceFailure(url: string): void {
  const host = hostOf(url);
  if (!host) return;
  failedHosts.set(host, deps.now());
  if (cached) cached.stale = true;
  generation += 1;
  inFlight = null;
}

// ---------------------------------------------------------------------------
// 测试口子
// ---------------------------------------------------------------------------

/** 替换探测 / 设置 / 地区 / 时钟（只传要换的），并清空缓存与失败记录。传 null 恢复默认。 */
export function __setNetSourcesDepsForTest(overrides: Partial<Deps> | null): void {
  deps = overrides ? { ...defaultDeps, ...overrides } : defaultDeps;
  __resetNetSourcesForTest();
}

export function __resetNetSourcesForTest(): void {
  cached = null;
  inFlight = null;
  generation += 1;
  failedHosts.clear();
}

/** 直接塞一份计划进缓存（mirror-download 等下游测试用，免得跑探测）。 */
export function __setSourcePlanForTest(plan: SourcePlan | null): void {
  __resetNetSourcesForTest();
  if (plan) cached = { plan: { ...plan, at: deps.now() }, sig: settingsSig(deps.settings()), stale: false };
}

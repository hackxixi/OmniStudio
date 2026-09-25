/**
 * 下载源路由：模型、Python 依赖、引擎二进制、Homebrew 等所有下载都从这里要「先走哪条路」。
 *
 * 判定不能只看地区：国内开着代理的用户直连 huggingface.co / github.com 往往比镜像还快，
 * 硬切镜像反而变慢。所以地区信号（时区 / 语言）只决定探测顺序与平局时的偏好，
 * 最终按实测「连得上 + 首包延迟」排序；设置里可以强制「国内加速 / 国际直连」。
 */

/** 用户设置 DOWNLOAD_REGION：auto = 探测决定；cn = 强制国内加速；global = 强制官方直连。 */
export type DownloadRegionSetting = "auto" | "cn" | "global";

export const DOWNLOAD_REGION_VALUES = ["auto", "cn", "global"] as const satisfies readonly DownloadRegionSetting[];

/** 探测目标的类别：决定它进哪张候选表（modelscope / homebrew 只展示、不排序）。 */
export type SourceKind = "hf" | "pypi" | "github" | "modelscope" | "homebrew";

/**
 * 单个源的探测结果（latencyMs = 首包耗时；不可达为 null）。
 * url 是「源本身」（HF 端点 / PyPI 索引 / GitHub 前缀，直连 GitHub 记作 https://github.com），
 * 不是探测时实际请求的那个小文件；kind / official 供界面分组、日志排障用。
 */
export type SourceProbe = {
  url: string;
  ok: boolean;
  latencyMs: number | null;
  kind?: SourceKind;
  official?: boolean;
};

/**
 * 路由结论。每个列表都是「按优先级排好的候选」，调用方依次尝试、失败换下一个；
 * 列表里永远保留官方源作为兜底（国内加速模式下排在最后）。
 */
export type SourcePlan = {
  /** 实际采用的模式：auto 探测后落到 cn / global。 */
  mode: "cn" | "global";
  /** 由谁决定：用户强制 / 探测 / 探测失败时按地区信号猜。 */
  decidedBy: "setting" | "probe" | "locale-guess";
  /** 地区信号（时区 Asia/Shanghai 等 / 语言 zh-CN）是否指向中国大陆。 */
  cnLocale: boolean;
  /** 模型市场 / 下载器的默认平台。 */
  modelSource: "modelscope" | "huggingface";
  /** Hugging Face 端点，按优先级（如 ["https://hf-mirror.com", "https://huggingface.co"]）。 */
  hfEndpoints: string[];
  /** PyPI simple 索引，按优先级。 */
  pypiIndexes: string[];
  /** GitHub 加速前缀（`<prefix><原始 URL>`）；空串 = 直连。按优先级。 */
  githubPrefixes: string[];
  /** Homebrew 镜像环境变量（global 模式为空对象）。 */
  homebrewEnv: Record<string, string>;
  /** 各源的探测明细（界面「检测结果」展示、日志排障用）。 */
  probes: SourceProbe[];
  /** 生成时间（ms），缓存判断用。 */
  at: number;
};

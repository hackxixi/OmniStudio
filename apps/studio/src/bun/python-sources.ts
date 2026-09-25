/**
 * Python 侧安装的下载源：PyPI 索引按下载源计划依次尝试、uv 本体的多链路安装。
 *
 * mflux / PaddleOCR / laya-mlx / mlx-lm·vLLM·SGLang 四处安装原来各写一份「默认源装一次，
 * 失败再换清华镜像」—— 国内用户每次都要先在 pypi.org 上耗到超时才轮到镜像。现在统一按
 * `SourcePlan.pypiIndexes` 的顺序来：第一次就用探测出来最快的索引，失败再换下一个。
 *
 * uv 本体也在这里装：`brew install uv` / `curl astral.sh | sh` 在国内都要连 GitHub，
 * 所以另备两条路 —— 经 GitHub 加速镜像下官方 release 的单文件包、或用已有的 Python
 * 从 PyPI 镜像 `pip install uv` —— 装进应用托管目录 `<dataDir>/engines/uv/bin`。
 */
import { chmodSync, copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from "fs";
import path from "path";

import type { SourcePlan } from "../shared/net-sources";
import type { AppLogSource } from "./app-log";
import { defaultCommandRunner, type CommandRunner } from "./command-runner";
import { fetchAssetFromSources } from "./mirror-download";
import { githubCandidates, reportSourceFailure, sourceEnv } from "./net-sources";
import { getDataDir } from "./paths";

const isWindows = process.platform === "win32";

// ---------------------------------------------------------------------------
// PyPI 索引
// ---------------------------------------------------------------------------

const OFFICIAL_PYPI = "https://pypi.org/simple";

/** 索引改由每次尝试的 `--index-url` 决定，环境变量里的默认索引要拿掉，免得和参数打架。 */
const INDEX_ENV_KEYS = ["PIP_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX_URL"];

function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

function isOfficialPypi(url: string): boolean {
  return trimSlash(url) === OFFICIAL_PYPI;
}

const PYPI_LABELS: [RegExp, string][] = [
  [/(^|\.)pypi\.org$/, "官方 PyPI 源"],
  [/aliyun\.com$/, "阿里云 PyPI 镜像"],
  [/tuna\.tsinghua\.edu\.cn$/, "清华 PyPI 镜像"],
  [/cloud\.tencent\.com$/, "腾讯云 PyPI 镜像"],
  [/huaweicloud\.com$/, "华为云 PyPI 镜像"],
  [/ustc\.edu\.cn$/, "中科大 PyPI 镜像"],
  [/sjtu\.edu\.cn$/, "上海交大 PyPI 镜像"],
  [/bfsu\.edu\.cn$/, "北外 PyPI 镜像"],
];

/** 日志里给人看的索引名（「用 阿里云 PyPI 镜像安装…」）。 */
export function pypiIndexLabel(url: string): string {
  let host = url;
  try {
    host = new URL(url).hostname;
  } catch {
    return `PyPI 源（${url}）`;
  }
  for (const [re, label] of PYPI_LABELS) if (re.test(host)) return label;
  return `PyPI 镜像（${host}）`;
}

/**
 * 这个索引对应的命令行参数。pip 与 `uv pip` 都认 `--index-url`（uv 新版把它标成
 * deprecated、推荐 `--default-index`，但老版本 uv 不认后者 —— 用户机器上的 uv 可能很旧）。
 * 官方源不加参数：等于工具自己的默认，用户自己配过的 pip.conf / uv.toml 照样生效。
 */
export function pypiIndexArgs(url: string): string[] {
  return isOfficialPypi(url) ? [] : ["--index-url", trimSlash(url)];
}

/** 计划里的索引（去重；空计划兜底官方源）。 */
export function planPypiIndexes(plan: SourcePlan): string[] {
  const out: string[] = [];
  for (const url of plan.pypiIndexes) {
    const u = trimSlash(url);
    if (u && !out.includes(u)) out.push(u);
  }
  return out.length ? out : [OFFICIAL_PYPI];
}

/**
 * 安装类子进程的附加环境：下载源计划的全部变量（HF_ENDPOINT、UV_PYTHON_INSTALL_MIRROR、
 * Homebrew 镜像…）减去默认索引（索引由每次尝试的参数决定）。调用方自己叠在 process.env 上。
 */
export function installEnv(plan: SourcePlan): Record<string, string> {
  const env = { ...sourceEnv(plan) };
  for (const key of INDEX_ENV_KEYS) delete env[key];
  return env;
}

/** Python 子进程（worker / 下载脚本）的附加环境：计划变量 + HF 端点候选（脚本失败时逐个换）。 */
export function pythonChildEnv(plan: SourcePlan): Record<string, string> {
  return { ...sourceEnv(plan), OMNI_HF_ENDPOINTS: plan.hfEndpoints.join(",") };
}

/**
 * 按计划的索引顺序装包：第一次就用最优的索引，失败（非零退出码）换下一个，直到成功或
 * 全部试完。返回最后一次的退出码。失败的索引上报给下载源路由，下次计划会把它往后排。
 */
export async function installFromIndexes(opts: {
  plan: SourcePlan;
  /** 装什么（进日志）：`mflux`、`laya-mlx`… */
  what: string;
  /** 用给定的索引参数跑一次安装命令，返回退出码。 */
  run: (indexArgs: string[], index: string) => Promise<number>;
  log: (line: string) => void;
}): Promise<{ code: number; index: string }> {
  const indexes = planPypiIndexes(opts.plan);
  let code = -1;
  let index = indexes[0]!;
  for (let i = 0; i < indexes.length; i++) {
    const prev = index;
    index = indexes[i]!;
    const label = pypiIndexLabel(index);
    opts.log(
      i === 0
        ? `用 ${label}安装 ${opts.what}…`
        : `${pypiIndexLabel(prev)}安装失败（退出码 ${code}），改用 ${label}重试…`,
    );
    code = await opts.run(pypiIndexArgs(index), index);
    if (code === 0) return { code, index };
    reportSourceFailure(index);
  }
  return { code, index };
}

// ---------------------------------------------------------------------------
// uv：解析与安装
// ---------------------------------------------------------------------------

/** 应用托管的 uv（`<dataDir>/engines/uv/bin/uv`）。 */
export function managedUvDir(): string {
  return getDataDir("engines", "uv");
}

export function managedUvBinDir(): string {
  return path.join(managedUvDir(), "bin");
}

export function managedUvPath(): string {
  return path.join(managedUvBinDir(), isWindows ? "uv.exe" : "uv");
}

/** 找 uv：托管目录优先（应用自己装的版本可复现），其次系统 PATH。 */
export function resolveUv(searchPath: string): string | null {
  const managed = managedUvPath();
  if (existsSync(managed)) return managed;
  return Bun.which("uv", { PATH: searchPath }) ?? null;
}

/** 官方 release 里当前平台的单文件包名；不在官方产物里的组合返回 null。 */
export function uvReleaseAsset(platform: string, arch: string): string | null {
  const cpu = arch === "arm64" ? "aarch64" : arch === "x64" ? "x86_64" : null;
  if (!cpu) return null;
  if (platform === "darwin") return `uv-${cpu}-apple-darwin.tar.gz`;
  if (platform === "linux") return `uv-${cpu}-unknown-linux-gnu.tar.gz`;
  if (platform === "win32") return `uv-${cpu}-pc-windows-msvc.zip`;
  return null;
}

export const UV_RELEASE_BASE = "https://github.com/astral-sh/uv/releases/latest/download";

export type UvInstallVia = "brew" | "script" | "github" | "pypi";

/** 装 uv 的顺序：国内先走镜像（单文件包 → PyPI 镜像 → 带镜像的 brew），海外沿用 brew / 官方脚本在前。 */
export function uvInstallOrder(plan: SourcePlan, hasBrew: boolean): UvInstallVia[] {
  if (plan.mode === "cn") return hasBrew ? ["github", "pypi", "brew"] : ["github", "pypi"];
  return hasBrew ? ["brew", "script", "github", "pypi"] : ["script", "github", "pypi"];
}

function findFile(root: string, name: string, depth = 3): string | null {
  let entries: string[];
  try {
    entries = readdirSync(root);
  } catch {
    return null;
  }
  if (entries.includes(name)) return path.join(root, name);
  if (depth <= 0) return null;
  for (const entry of entries) {
    const child = path.join(root, entry);
    try {
      if (!statSync(child).isDirectory()) continue;
    } catch {
      continue;
    }
    const found = findFile(child, name, depth - 1);
    if (found) return found;
  }
  return null;
}

export type UvDownloader = (opts: {
  urls: string[];
  dest: string;
  what: string;
  accept: (file: string) => Promise<string | null>;
}) => Promise<{ ok: boolean; error?: string }>;

export type UvInstallDeps = {
  plan: SourcePlan;
  log: (line: string) => void;
  /** 找 brew / python 用的 PATH。 */
  searchPath: string;
  source: AppLogSource;
  runner?: CommandRunner;
  platform?: string;
  arch?: string;
  /** 测试注入：GitHub 多链路下载（默认 mirror-download 的 fetchAssetFromSources）。 */
  download?: UvDownloader;
  /** 测试注入：找任意一个能跑 `-m venv` 的 Python（PyPI 那条路用）。 */
  findAnyPython?: () => string | null;
  /** 测试注入：找 brew。 */
  findBrew?: () => string | null;
};

export type UvInstallResult = { ok: boolean; path?: string; via?: UvInstallVia; error?: string };

const defaultDownloader =
  (source: AppLogSource): UvDownloader =>
  async ({ urls, dest, what, accept }) => {
    const res = await fetchAssetFromSources({
      urls,
      dest,
      what,
      source,
      accept: (file, head, bytes) => {
        if (bytes < 1_000_000) return `文件过小（${bytes} 字节），不是 uv 发布包`;
        const gz = head[0] === 0x1f && head[1] === 0x8b;
        const zip = head[0] === 0x50 && head[1] === 0x4b;
        if (!gz && !zip) return "不是有效的压缩包（可能下到了代理错误页）";
        return accept(file);
      },
    });
    return res.ok ? { ok: true } : { ok: false, error: res.error };
  };

/**
 * 任意一个 Python 3（只用来建个临时 venv 跑 `pip install uv`，版本不挑）。
 * macOS 的 /usr/bin/python3 在没装命令行工具时是个会弹安装窗的桩，先确认 CLT 在。
 */
function defaultFindAnyPython(searchPath: string, runner: CommandRunner): string | null {
  for (const name of ["python3.13", "python3.12", "python3.11", "python3.10", "python3"]) {
    const bin = Bun.which(name, { PATH: searchPath });
    if (!bin) continue;
    if (process.platform === "darwin" && bin === "/usr/bin/python3") {
      if (runner.run(["xcode-select", "-p"], 5_000).code !== 0) continue;
    }
    return bin;
  }
  return null;
}

/** 单文件包：解包 → 取出 uv / uvx → 放进托管目录 → `uv --version` 自证能跑。 */
async function installUvFromGithub(deps: UvInstallDeps, runner: CommandRunner): Promise<string | null> {
  const asset = uvReleaseAsset(deps.platform ?? process.platform, deps.arch ?? process.arch);
  if (!asset) {
    deps.log("当前平台没有 uv 官方构建，跳过 GitHub 下载");
    return null;
  }
  const dir = managedUvDir();
  mkdirSync(dir, { recursive: true });
  const staging = path.join(dir, `.staging-${process.pid}`);
  const dest = path.join(dir, `.${asset}`);
  const urls = githubCandidates(`${UV_RELEASE_BASE}/${asset}`, deps.plan);
  deps.log(`从 GitHub 下载 uv（${asset}，${urls.length} 条链路）…`);
  const download = deps.download ?? defaultDownloader(deps.source);
  const exe = isWindows ? "uv.exe" : "uv";
  try {
    const res = await download({
      urls,
      dest,
      what: "uv",
      accept: async (file) => {
        rmSync(staging, { recursive: true, force: true });
        mkdirSync(staging, { recursive: true });
        const tar = runner.run(["tar", "-xf", file, "-C", staging], 60_000);
        if (tar.code !== 0) return `解包失败：${(tar.stderr || tar.stdout).slice(-200)}`;
        const uv = findFile(staging, exe);
        if (!uv) return "包里没有 uv 可执行文件";
        mkdirSync(managedUvBinDir(), { recursive: true });
        for (const name of [exe, isWindows ? "uvx.exe" : "uvx"]) {
          const src = findFile(staging, name);
          if (!src) continue;
          const target = path.join(managedUvBinDir(), name);
          copyFileSync(src, target);
          try {
            chmodSync(target, 0o755);
          } catch {
            // Windows 无 POSIX 权限
          }
        }
        const check = runner.run([managedUvPath(), "--version"], 15_000);
        if (check.code !== 0) {
          rmSync(managedUvBinDir(), { recursive: true, force: true });
          return `uv 无法运行：${(check.stderr || check.stdout).slice(-200)}`;
        }
        return null;
      },
    });
    if (!res.ok) {
      deps.log(`GitHub 下载 uv 失败：${res.error ?? "未知错误"}`);
      return null;
    }
    return existsSync(managedUvPath()) ? managedUvPath() : null;
  } finally {
    rmSync(staging, { recursive: true, force: true });
    rmSync(dest, { force: true });
  }
}

/** PyPI 那条路：临时 venv 里 `pip install uv`（索引按计划轮换），把 uv 二进制拷进托管目录。 */
async function installUvFromPypi(deps: UvInstallDeps, runner: CommandRunner): Promise<string | null> {
  const python = (deps.findAnyPython ?? (() => defaultFindAnyPython(deps.searchPath, runner)))();
  if (!python) {
    deps.log("没有可用的 Python，跳过 PyPI 安装 uv");
    return null;
  }
  const venv = path.join(managedUvDir(), ".pip-venv");
  rmSync(venv, { recursive: true, force: true });
  mkdirSync(managedUvDir(), { recursive: true });
  const onLine = (line: string) => deps.log(line);
  const env = installEnv(deps.plan);
  try {
    const venvCmd = [python, "-m", "venv", venv];
    deps.log(`$ ${venvCmd.join(" ")}`);
    if ((await runner.runStreaming(venvCmd, onLine, { env })) !== 0) return null;
    const pip = isWindows ? path.join(venv, "Scripts", "pip.exe") : path.join(venv, "bin", "pip");
    const { code } = await installFromIndexes({
      plan: deps.plan,
      what: "uv",
      log: deps.log,
      run: (indexArgs) => {
        const cmd = [pip, "install", "--disable-pip-version-check", "uv", ...indexArgs];
        deps.log(`$ ${cmd.join(" ")}`);
        return runner.runStreaming(cmd, onLine, { env });
      },
    });
    if (code !== 0) return null;
    const exe = isWindows ? "uv.exe" : "uv";
    const built = isWindows ? path.join(venv, "Scripts", exe) : path.join(venv, "bin", exe);
    if (!existsSync(built)) return null;
    // uv 的 wheel 里就是那个静态二进制，拷出来即可独立使用，临时 venv 随后删掉。
    mkdirSync(managedUvBinDir(), { recursive: true });
    copyFileSync(built, managedUvPath());
    try {
      chmodSync(managedUvPath(), 0o755);
    } catch {
      // Windows 无 POSIX 权限
    }
    return managedUvPath();
  } finally {
    rmSync(venv, { recursive: true, force: true });
  }
}

/**
 * 装 uv：按 `uvInstallOrder` 逐条尝试，任何一条让 `resolveUv` 找得到 uv 就算成功。
 * brew 带上计划里的 Homebrew 镜像变量；官方脚本只在海外模式尝试（它自己也要连 GitHub）。
 */
export async function installUv(deps: UvInstallDeps): Promise<UvInstallResult> {
  const runner = deps.runner ?? defaultCommandRunner;
  const brew = (deps.findBrew ?? (() => Bun.which("brew", { PATH: deps.searchPath }) ?? null))();
  const order = uvInstallOrder(deps.plan, !!brew);
  const errors: string[] = [];
  const onLine = (line: string) => deps.log(line);
  for (const via of order) {
    try {
      if (via === "brew" || via === "script") {
        const cmd =
          via === "brew"
            ? [brew!, "install", "uv"]
            : ["/bin/sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"];
        deps.log(`$ ${cmd.join(" ")}`);
        const env = via === "brew" ? { ...installEnv(deps.plan), ...deps.plan.homebrewEnv } : installEnv(deps.plan);
        const code = await runner.runStreaming(cmd, onLine, { env });
        const found = resolveUv(deps.searchPath);
        if (code === 0 && found) return { ok: true, path: found, via };
        errors.push(`${via}：退出码 ${code}`);
        deps.log(`${via === "brew" ? "brew install uv" : "官方安装脚本"}失败（退出码 ${code}），换下一种方式…`);
        continue;
      }
      const found =
        via === "github" ? await installUvFromGithub(deps, runner) : await installUvFromPypi(deps, runner);
      if (found) return { ok: true, path: found, via };
      errors.push(`${via}：失败`);
    } catch (e) {
      errors.push(`${via}：${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return { ok: false, error: errors.join("；") || "没有可用的安装方式" };
}

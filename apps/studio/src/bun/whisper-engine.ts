import { chmodSync, existsSync, mkdirSync, readdirSync, renameSync, rmSync } from "fs";
import path from "path";
import { WHISPER_CPP_RELEASE_TAG, WHISPER_CPP_REPO } from "../shared/whispercpp";
import { getDataDir } from "./paths";
import { removeManifest, writeManifest } from "./install-manifest";
import { fetchAssetFromSources, githubReleaseUrls, officialWithMirrors } from "./mirror-download";
import { getSourcePlan } from "./net-sources";
import type { SourcePlan } from "../shared/net-sources";

/**
 * whisper.cpp 本地识别引擎（whisper-cli / whisper-server）的一键安装。
 *
 * 上游新版 Release 不再发布 macOS CLI 二进制（只有 Linux/Windows 资产），
 * macOS 因此走 conda-forge 官方预编译包（whisper.cpp + 依赖 llvm-openmp / libcxx，
 * 自带 @rpath 可迁移布局），其余平台走 GitHub Release 资产。统一解压到
 * userData/engines/whispercpp/current/，识别时优先于 PATH（brew 等）使用。
 */

export type WhisperEngineInfo = {
  installed: boolean;
  version: string | null;
  binaryPath: string | null;
  /** 当前平台是否支持一键安装。 */
  supported: boolean;
};

function getEnginesDir(): string {
  return getDataDir("engines", "whispercpp");
}

/** 引擎安装根目录：conda 布局为 bin/ + lib/；Linux/Windows 资产直接在根下。 */
function getEngineRoot(): string {
  return path.join(getEnginesDir(), "current");
}

function getSearchPath(): string {
  const home = process.env.HOME ?? "";
  const extra = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    home ? `${home}/.local/bin` : "",
    home ? `${home}/bin` : "",
  ];
  const current = process.env.PATH ?? "";
  return [...extra, current].join(":");
}

async function findOnPath(name: string): Promise<string | null> {
  const p = Bun.which(name, { PATH: getSearchPath() });
  return p ?? null;
}

/** 应用内置的 whisper 二进制（userData/engines/whispercpp/current/bin 或根目录）。 */
function bundledBinary(name: "whisper-cli" | "whisper-server"): string | null {
  for (const dir of [path.join(getEngineRoot(), "bin"), getEngineRoot()]) {
    const p = path.join(dir, name);
    if (existsSync(p)) return p;
  }
  return null;
}

/** 找到 whisper 二进制：优先应用内置引擎，其次 PATH（brew 等）。 */
export async function resolveWhisperBinary(
  name: "whisper-cli" | "whisper-server",
): Promise<string | null> {
  try {
    const bundled = bundledBinary(name);
    if (bundled) return bundled;
  } catch {}
  return await findOnPath(name);
}

export async function getWhisperEngineInfo(): Promise<WhisperEngineInfo> {
  const [cli, server] = await Promise.all([
    resolveWhisperBinary("whisper-cli"),
    resolveWhisperBinary("whisper-server"),
  ]);
  let version: string | null = null;
  try {
    const marker = path.join(getEngineRoot(), "VERSION");
    if (existsSync(marker)) version = (await Bun.file(marker).text()).trim() || null;
  } catch {
    // ignore
  }
  return {
    installed: !!cli || !!server,
    version,
    binaryPath: cli ?? server ?? null,
    supported: process.platform === "darwin" || releaseAssetName() !== null,
  };
}

// ---------------------------------------------------------------------------
// 安装收尾
// ---------------------------------------------------------------------------

/** 保证二进制可执行并写入版本标记。 */
async function finalize(engineRoot: string, version: string): Promise<void> {
  for (const name of ["whisper-cli", "whisper-server"] as const) {
    const p = bundledBinary(name);
    if (p) {
      try {
        chmodSync(p, 0o755);
      } catch {
        // Windows 无 POSIX 权限
      }
    }
  }
  try {
    await Bun.write(path.join(engineRoot, "VERSION"), version);
  } catch {
    // ignore
  }
}

function cleanupStaging(staging: string): void {
  rmSync(staging, { recursive: true, force: true });
}

// ---------------------------------------------------------------------------
// macOS：conda-forge 官方预编译包
// ---------------------------------------------------------------------------

type CondaFile = { pkg: string; version: string; basename: string } | null;

/** conda-forge 的平台子目录（本引擎只在 macOS 走 conda 包）。 */
function condaSubdir(): string {
  return process.platform === "darwin" ? (process.arch === "arm64" ? "osx-arm64" : "osx-64") : "";
}

/** 锚定 conda-forge 的 whisper.cpp 及其依赖（llvm-openmp / libcxx），取当前平台最新。 */
async function fetchCondaFile(pkg: string, timeoutMs: number): Promise<CondaFile> {
  try {
    const res = await fetch(`https://api.anaconda.org/package/conda-forge/${pkg}`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    const json = (await res.json()) as {
      latest_version?: string;
      files?: { basename?: string; version?: string }[];
    };
    const version = json.latest_version;
    if (!version || !Array.isArray(json.files)) return null;
    const subdir = condaSubdir();
    const file = json.files.find(
      (f) => f.version === version && f.basename?.startsWith(`${subdir}/`) && f.basename.endsWith(".conda"),
    );
    if (!file?.basename) return null;
    return { pkg, version, basename: file.basename };
  } catch {
    return null;
  }
}

/** conda 版本号比较（按 `.` / `_` / `-` 分段，数字段按数值、其余按字典序），返回 -1 / 0 / 1。 */
export function compareCondaVersion(a: string, b: string): number {
  const pa = a.split(/[._-]/);
  const pb = b.split(/[._-]/);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i] ?? "0";
    const y = pb[i] ?? "0";
    const nx = /^\d+$/.test(x) ? Number(x) : NaN;
    const ny = /^\d+$/.test(y) ? Number(y) : NaN;
    if (!Number.isNaN(nx) && !Number.isNaN(ny)) {
      if (nx !== ny) return nx < ny ? -1 : 1;
    } else if (x !== y) {
      return x < y ? -1 : 1;
    }
  }
  return 0;
}

type RepodataEntry = { name?: string; version?: string; build?: string; build_number?: number; timestamp?: number };

/**
 * 从频道索引（repodata）里挑某个包的最新 `.conda`：版本最高 → build 号最大 → 非 debug 构建 →
 * 最新上传。返回的 basename 带子目录前缀（与 api.anaconda.org 的 files[].basename 同形）。
 */
export function pickFromRepodata(
  repodata: { packages?: Record<string, RepodataEntry>; "packages.conda"?: Record<string, RepodataEntry> },
  pkg: string,
  subdir: string,
): CondaFile {
  let best: { file: string; entry: RepodataEntry } | null = null;
  for (const [file, entry] of Object.entries(repodata["packages.conda"] ?? {})) {
    if (entry.name !== pkg || !entry.version) continue;
    if (!best) {
      best = { file, entry };
      continue;
    }
    const b = best.entry;
    const byVersion = compareCondaVersion(entry.version, b.version!);
    const byBuild = (entry.build_number ?? 0) - (b.build_number ?? 0);
    const debugRank = Number(!(entry.build ?? "").startsWith("debug")) - Number(!(b.build ?? "").startsWith("debug"));
    const byTime = (entry.timestamp ?? 0) - (b.timestamp ?? 0);
    const better = byVersion !== 0 ? byVersion > 0 : byBuild !== 0 ? byBuild > 0 : debugRank !== 0 ? debugRank > 0 : byTime > 0;
    if (better) best = { file, entry };
  }
  if (!best) return null;
  return { pkg, version: best.entry.version!, basename: `${subdir}/${best.file}` };
}

/**
 * api.anaconda.org 不可达时的兜底：直接读频道索引 `current_repodata.json`（只含各包最新版，
 * osx-arm64 约 30MB）。国内先读清华 / 上交镜像，海外先读 conda.anaconda.org（CDN）。
 */
function repodataUrls(subdir: string, plan: SourcePlan): string[] {
  const official = `https://conda.anaconda.org/conda-forge/${subdir}/current_repodata.json`;
  const mirrors = CONDA_MIRRORS.map((m) => `${m}/${subdir}/current_repodata.json`);
  return plan.mode === "cn" ? [...mirrors, official] : [official, ...mirrors];
}

async function fetchCondaFilesFromRepodata(
  pkgs: string[],
  plan: SourcePlan,
  fetchImpl: typeof fetch = fetch,
): Promise<Map<string, CondaFile>> {
  const subdir = condaSubdir();
  const out = new Map<string, CondaFile>();
  for (const url of repodataUrls(subdir, plan)) {
    try {
      const res = await fetchImpl(url, { signal: AbortSignal.timeout(180_000) });
      if (!res.ok) continue;
      const repodata = (await res.json()) as Parameters<typeof pickFromRepodata>[0];
      for (const pkg of pkgs) out.set(pkg, pickFromRepodata(repodata, pkg, subdir));
      if (pkgs.every((pkg) => out.get(pkg))) return out;
    } catch {
      // 换下一个索引源
    }
  }
  return out;
}

/**
 * 三个包的最新文件：先问 api.anaconda.org（国内给短超时，它在 AWS 上时通时不通），
 * 有任何一个问不到就整体改读频道索引。
 */
export async function resolveCondaFiles(
  pkgs: string[],
  plan: SourcePlan,
  deps: {
    fromApi?: (pkg: string, timeoutMs: number) => Promise<CondaFile>;
    fromRepodata?: (pkgs: string[], plan: SourcePlan) => Promise<Map<string, CondaFile>>;
  } = {},
): Promise<CondaFile[]> {
  const fromApi = deps.fromApi ?? fetchCondaFile;
  const fromRepodata = deps.fromRepodata ?? ((p: string[], pl: SourcePlan) => fetchCondaFilesFromRepodata(p, pl));
  const timeoutMs = plan.mode === "cn" ? 10_000 : 30_000;
  const viaApi = await Promise.all(pkgs.map((pkg) => fromApi(pkg, timeoutMs)));
  if (viaApi.every(Boolean)) return viaApi;
  const viaIndex = await fromRepodata(pkgs, plan);
  return pkgs.map((pkg, i) => viaApi[i] ?? viaIndex.get(pkg) ?? null);
}

/**
 * conda-forge 的国内镜像（频道根）。镜像的目录结构是 `<频道>/<子目录>/<文件名>`
 * （与 conda.anaconda.org 相同），不是 api.anaconda.org 的 `<包>/<版本>/<子目录>/<文件名>`。
 */
const CONDA_MIRRORS = [
  "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge",
  "https://mirror.sjtu.edu.cn/anaconda/cloud/conda-forge",
] as const;

/** conda 包下载地址（官方 + CDN + 国内镜像回退）。 */
export function condaUrls(f: NonNullable<CondaFile>): string[] {
  const { pkg, version, basename } = f;
  return [
    `https://api.anaconda.org/download/conda-forge/${pkg}/${version}/${basename}`,
    `https://conda.anaconda.org/conda-forge/${basename}`,
    ...CONDA_MIRRORS.map((m) => `${m}/${basename}`),
  ];
}

/** 解压 .conda（zip 内含 pkg-*.tar.zst 数据包）到 staging。 */
async function extractCondaArchive(tmp: string, staging: string): Promise<void> {
  const unzip = Bun.spawnSync(["unzip", "-q", "-o", tmp, "-d", staging], {
    stdout: "pipe",
    stderr: "pipe",
  });
  if (unzip.exitCode !== 0) {
    throw new Error(`conda 解压失败：${unzip.stderr.toString().slice(-200)}`);
  }
  const data = readdirSync(staging).find((n) => n.startsWith("pkg-") && n.endsWith(".tar.zst"));
  if (!data) throw new Error("conda 包内缺少 pkg 数据");
  // bsdtar 可自动识别 zstd；个别系统不支持时再显式加 --zstd。
  let tar = Bun.spawnSync(["tar", "-xf", path.join(staging, data), "-C", staging], {
    stdout: "pipe",
    stderr: "pipe",
  });
  if (tar.exitCode !== 0) {
    tar = Bun.spawnSync(["tar", "--zstd", "-xf", path.join(staging, data), "-C", staging], {
      stdout: "pipe",
      stderr: "pipe",
    });
  }
  if (tar.exitCode !== 0) {
    throw new Error(`conda 数据解压失败：${tar.stderr.toString().slice(-200)}`);
  }
  for (const n of readdirSync(staging)) {
    if (n.endsWith(".tar.zst") || n === "metadata.json") {
      rmSync(path.join(staging, n), { force: true });
    }
  }
}

/**
 * 下载并合并 whisper.cpp + llvm-openmp + libcxx 三个 conda 包到引擎目录。
 * （libomp / libc++ 是 whisper.cpp 的运行依赖，合并后即可脱离 conda 使用。）
 */
async function downloadCondaEngine(): Promise<{ ok: boolean; error?: string; version?: string }> {
  const [whisper, omp, cxx] = await resolveCondaFiles(
    ["whisper.cpp", "llvm-openmp", "libcxx"],
    await getSourcePlan(),
  );
  for (const f of [whisper, omp, cxx]) {
    if (!f) {
      return { ok: false, error: "无法获取 whisper.cpp 的 macOS 预编译包（conda-forge 不可达）" };
    }
  }
  const cfgs = [whisper!, omp!, cxx!];
  const version = whisper!.version;

  mkdirSync(getEnginesDir(), { recursive: true });
  const staging = path.join(getEnginesDir(), `.staging-${process.pid}`);
  cleanupStaging(staging);
  mkdirSync(staging, { recursive: true });

  for (const f of cfgs) {
    const basename = f.basename.split("/").pop()!;
    const tmp = path.join(getEnginesDir(), `.${basename}`);
    const condaAll = condaUrls(f);
    const res = await fetchAssetFromSources({
      urls: await officialWithMirrors(condaAll[0]!, condaAll.slice(1)),
      dest: tmp,
      what: `whisper.cpp 依赖包 ${f.pkg}`,
      source: "asr",
      accept: async (file) => {
        try {
          await extractCondaArchive(file, staging);
          return null;
        } catch (e) {
          // 解压失败：铺白 staging，别让下一个源读到上一个源的脏数据。
          rmSync(staging, { recursive: true, force: true });
          mkdirSync(staging, { recursive: true });
          return e instanceof Error ? e.message : String(e);
        }
      },
    });
    rmSync(tmp, { force: true });
    if (!res.ok) {
      cleanupStaging(staging);
      return { ok: false, error: `whisper.cpp 引擎安装失败（${f.pkg}）：${res.error}` };
    }
  }

  const engineRoot = getEngineRoot();
  rmSync(engineRoot, { recursive: true, force: true });
  try {
    renameSync(staging, engineRoot);
  } catch (e) {
    cleanupStaging(staging);
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  await finalize(engineRoot, version);
  writeManifest(getEnginesDir(), {
    engine: "whisper.cpp",
    version,
    platform: process.platform,
    arch: process.arch,
    steps: 2,
  });
  return bundledBinary("whisper-cli")
    ? { ok: true, version }
    : { ok: false, error: "安装后未找到 whisper-cli" };
}

// ---------------------------------------------------------------------------
// Linux / Windows：GitHub Release 资产
// ---------------------------------------------------------------------------

/** GitHub Release 资产文件名（Linux / Windows）。 */
function releaseAssetName(): string | null {
  if (process.platform === "linux" && process.arch === "x64") return "whisper-bin-ubuntu-x64.tar.gz";
  if (process.platform === "linux" && process.arch === "arm64") return "whisper-bin-ubuntu-arm64.tar.gz";
  if (process.platform === "win32" && process.arch === "x64") return "whisper-bin-x64.zip";
  return null;
}

/** 在解压目录里定位“内容根”（Linux 资产顶层即含 whisper-cli 的目录）。 */
function findContentRoot(root: string): string | null {
  const stack = [{ dir: root, depth: 0 }];
  while (stack.length > 0) {
    const { dir, depth } = stack.pop()!;
    try {
      const entries = readdirSync(dir, { withFileTypes: true });
      if (entries.some((e) => e.isFile() && e.name === "whisper-cli")) return dir;
      if (depth < 4) {
        for (const e of entries) {
          if (e.isDirectory()) stack.push({ dir: path.join(dir, e.name), depth: depth + 1 });
        }
      }
    } catch {
      // unreadable entry — skip
    }
  }
  return null;
}

async function installGitHubRelease(tmp: string, version: string): Promise<{ ok: boolean; error?: string }> {
  const engineRoot = getEngineRoot();
  rmSync(engineRoot, { recursive: true, force: true });
  mkdirSync(engineRoot, { recursive: true });

  const staging = path.join(getEnginesDir(), `.staging-${process.pid}`);
  cleanupStaging(staging);
  mkdirSync(staging, { recursive: true });
  try {
    // bsdtar 同时支持 gzip 与 zip；macOS / Windows 10+ 均自带。
    const tar = Bun.spawnSync(["tar", "-xf", tmp, "-C", staging], {
      stdout: "pipe",
      stderr: "pipe",
    });
    if (tar.exitCode !== 0) throw new Error(tar.stderr.toString().slice(-200));
  } catch (e) {
    cleanupStaging(staging);
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }

  const contentRoot = findContentRoot(staging);
  if (!contentRoot) {
    cleanupStaging(staging);
    return { ok: false, error: "解压后未找到 whisper-cli" };
  }

  try {
    renameSync(contentRoot, engineRoot);
  } catch (e) {
    cleanupStaging(staging);
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  cleanupStaging(staging);
  await finalize(engineRoot, version);
  return bundledBinary("whisper-cli") ? { ok: true } : { ok: false, error: "安装后未找到 whisper-cli" };
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------

/**
 * 一键安装 whisper.cpp 引擎（whisper-cli / whisper-server 均含）：
 * - macOS：conda-forge 预编译包（whisper.cpp + llvm-openmp + libcxx，多镜像回退）
 * - Linux / Windows：GitHub Release 资产（多镜像回退）
 * 已安装（应用内置或 PATH 已有）时直接返回成功；`upgrade` 时跳过这一步重装一遍
 * （macOS 因此能拿到 conda-forge 上的新版本，其余平台等于修复一份被破坏的安装）。
 */
export async function downloadWhisperEngine(
  options: { upgrade?: boolean } = {},
): Promise<{ ok: boolean; error?: string; version?: string }> {
  // 动任何文件之前先清掉上次的 manifest（与安装完成时的写入配对）。
  if (!removeManifest(getEnginesDir())) {
    const error = `无法清除上次的安装记录，请检查 ${getEnginesDir()} 是否被占用或只读`;
    return { ok: false, error };
  }
  if (!options.upgrade) {
    const installed = await resolveWhisperBinary("whisper-cli");
    if (installed) {
      return { ok: true, version: (await getWhisperEngineInfo()).version ?? undefined };
    }
  }

  if (process.platform === "darwin") {
    return await downloadCondaEngine();
  }

  const asset = releaseAssetName();
  if (!asset) {
    return {
      ok: false,
      error: `当前平台（${process.platform}/${process.arch}）暂不支持自动安装 whisper.cpp，请手动安装 whisper-cli / whisper-server 并加入 PATH`,
    };
  }

  mkdirSync(getEnginesDir(), { recursive: true });
  const tmp = path.join(getEnginesDir(), `.engine-${asset}`);
  const res = await fetchAssetFromSources({
    urls: await githubReleaseUrls(WHISPER_CPP_REPO, WHISPER_CPP_RELEASE_TAG, asset),
    dest: tmp,
    what: "whisper.cpp 引擎",
    source: "asr",
    accept: async (file) => {
      const r = await installGitHubRelease(file, WHISPER_CPP_RELEASE_TAG);
      return r.ok ? null : (r.error ?? "安装失败");
    },
  });
  rmSync(tmp, { force: true });
  if (!res.ok) return { ok: false, error: res.error };
  writeManifest(getEnginesDir(), {
    engine: "whisper.cpp",
    version: WHISPER_CPP_RELEASE_TAG,
    platform: process.platform,
    arch: process.arch,
    steps: 2,
  });
  return { ok: true, version: WHISPER_CPP_RELEASE_TAG };
}

/** 删除应用内置的引擎（PATH 上的不受影响）：整个 `<engines>/whispercpp` 一起删，暂存残留一并清掉。 */
export function deleteWhisperEngine(): void {
  rmSync(getEnginesDir(), { recursive: true, force: true });
}

/** 托管目录（引擎管理页展示「占用 / 路径」与卸载的目标）。 */
export function whisperEngineDir(): string {
  return getEnginesDir();
}

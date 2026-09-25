import { afterEach, describe, expect, test } from "bun:test";
import type { SourceProbe } from "../shared/net-sources";
import {
  GITHUB_OFFICIAL,
  GITHUB_PREFIX_MIRRORS,
  HF_OFFICIAL,
  MODELSCOPE_URL,
  PYPI_OFFICIAL,
  __setNetSourcesDepsForTest,
  decidePlan,
  detectCnLocale,
  getSourcePlan,
  githubCandidates,
  peekSourcePlan,
  reportSourceFailure,
  sourceEnv,
  type ProbeFn,
} from "./net-sources";

const HF_MIRROR = "https://hf-mirror.com";
const ALIYUN = "https://mirrors.aliyun.com/pypi/simple";
const TUNA = "https://pypi.tuna.tsinghua.edu.cn/simple";
const USTC = "https://mirrors.ustc.edu.cn/homebrew-bottles";

type Lat = number | null; // null = 不可达

/** 按「源 → 延迟」造一份完整探测明细（没列出的源按 100ms 可达）。 */
function probes(over: Record<string, Lat> = {}): SourceProbe[] {
  const all: [string, SourceProbe["kind"], boolean][] = [
    [HF_OFFICIAL, "hf", true],
    [HF_MIRROR, "hf", false],
    [MODELSCOPE_URL, "modelscope", false],
    [PYPI_OFFICIAL, "pypi", true],
    [ALIYUN, "pypi", false],
    [TUNA, "pypi", false],
    [GITHUB_OFFICIAL, "github", true],
    ...GITHUB_PREFIX_MIRRORS.map((p): [string, SourceProbe["kind"], boolean] => [p, "github", false]),
    [USTC, "homebrew", false],
  ];
  return all.map(([url, kind, official]) => {
    const lat = url in over ? over[url]! : 100;
    return { url, kind, official, ok: lat !== null, latencyMs: lat };
  });
}

afterEach(() => {
  __setNetSourcesDepsForTest(null);
});

describe("地区信号", () => {
  test("大陆时区算", () => {
    for (const tz of ["Asia/Shanghai", "Asia/Chongqing", "Asia/Harbin", "Asia/Urumqi", "Asia/Kashgar"]) {
      expect(detectCnLocale({ timeZone: tz })).toBe(true);
    }
  });
  test("港台 / 其他时区不算", () => {
    expect(detectCnLocale({ timeZone: "Asia/Hong_Kong" })).toBe(false);
    expect(detectCnLocale({ timeZone: "Asia/Taipei" })).toBe(false);
    expect(detectCnLocale({ timeZone: "America/Los_Angeles", langs: ["en_US.UTF-8"] })).toBe(false);
  });
  test("zh_CN / zh-Hans-CN 语言算，zh_TW / zh_HK / zh-Hant 不算", () => {
    expect(detectCnLocale({ langs: ["zh_CN.UTF-8"] })).toBe(true);
    expect(detectCnLocale({ langs: [undefined, "zh-Hans-CN"] })).toBe(true);
    expect(detectCnLocale({ langs: ["zh_Hans_CN"] })).toBe(true);
    expect(detectCnLocale({ langs: ["zh_TW.UTF-8", "zh_HK", "zh-Hant-TW", "zh-Hans-SG"] })).toBe(false);
  });
});

describe("决策（auto）", () => {
  test("官方全通且快：官方直连，官方排第一", () => {
    const p = decidePlan({ region: "auto", cnLocale: true, probes: probes() });
    expect(p.mode).toBe("global");
    expect(p.decidedBy).toBe("probe");
    expect(p.modelSource).toBe("huggingface");
    expect(p.hfEndpoints[0]).toBe(HF_OFFICIAL);
    expect(p.pypiIndexes[0]).toBe(PYPI_OFFICIAL);
    expect(p.githubPrefixes[0]).toBe("");
    expect(p.homebrewEnv).toEqual({});
  });

  test("官方 HF 不通：国内加速，镜像在前、官方兜底", () => {
    const p = decidePlan({ region: "auto", cnLocale: false, probes: probes({ [HF_OFFICIAL]: null }) });
    expect(p.mode).toBe("cn");
    expect(p.modelSource).toBe("modelscope");
    expect(p.hfEndpoints).toEqual([HF_MIRROR, HF_OFFICIAL]);
    expect(p.pypiIndexes[p.pypiIndexes.length - 1]).toBe(PYPI_OFFICIAL);
    expect(p.githubPrefixes[p.githubPrefixes.length - 1]).toBe("");
  });

  test("官方 GitHub 不通：同样国内加速", () => {
    expect(decidePlan({ region: "auto", cnLocale: false, probes: probes({ [GITHUB_OFFICIAL]: null }) }).mode).toBe("cn");
  });

  test("开着代理的大陆用户：官方只是略慢，不换源", () => {
    const p = decidePlan({
      region: "auto",
      cnLocale: true,
      probes: probes({ [HF_OFFICIAL]: 700, [HF_MIRROR]: 200, [GITHUB_OFFICIAL]: 600 }),
    });
    expect(p.mode).toBe("global");
  });

  test("大陆地区 + 镜像明显更快（>2× 且 >800ms）：国内加速", () => {
    const p = decidePlan({ region: "auto", cnLocale: true, probes: probes({ [HF_OFFICIAL]: 1500, [HF_MIRROR]: 200 }) });
    expect(p.mode).toBe("cn");
    expect(p.decidedBy).toBe("probe");
  });

  test("非大陆地区时镜像更快也不换（海外用户走镜像多半更慢）", () => {
    const p = decidePlan({ region: "auto", cnLocale: false, probes: probes({ [HF_OFFICIAL]: 1500, [HF_MIRROR]: 200 }) });
    expect(p.mode).toBe("global");
  });

  test("一个都探不通：按地区猜", () => {
    const allDown = probes(Object.fromEntries(probes().map((x) => [x.url, null])));
    expect(decidePlan({ region: "auto", cnLocale: true, probes: allDown })).toMatchObject({ mode: "cn", decidedBy: "locale-guess" });
    expect(decidePlan({ region: "auto", cnLocale: false, probes: [] })).toMatchObject({ mode: "global", decidedBy: "locale-guess" });
  });

  test("同类镜像按可达 → 延迟排序", () => {
    const p = decidePlan({
      region: "auto",
      cnLocale: false,
      probes: probes({
        [GITHUB_OFFICIAL]: null,
        [ALIYUN]: 400,
        [TUNA]: 50,
        "https://gh-proxy.com/": null,
        "https://ghfast.top/": 300,
        "https://ghproxy.net/": 90,
      }),
    });
    expect(p.pypiIndexes).toEqual([TUNA, ALIYUN, PYPI_OFFICIAL]);
    expect(p.githubPrefixes).toEqual(["https://ghproxy.net/", "https://ghfast.top/", "https://gh-proxy.com/", ""]);
  });

  test("global 模式里官方不可达时降到最后", () => {
    const p = decidePlan({ region: "global", cnLocale: false, probes: probes({ [PYPI_OFFICIAL]: null }) });
    expect(p.pypiIndexes[p.pypiIndexes.length - 1]).toBe(PYPI_OFFICIAL);
    expect(p.hfEndpoints[0]).toBe(HF_OFFICIAL);
  });
});

describe("设置强制与覆盖", () => {
  test("强制 cn：即使官方又快又通也走镜像", () => {
    const p = decidePlan({ region: "cn", cnLocale: false, probes: probes({ [HF_OFFICIAL]: 10 }) });
    expect(p).toMatchObject({ mode: "cn", decidedBy: "setting", modelSource: "modelscope" });
    expect(p.hfEndpoints[0]).toBe(HF_MIRROR);
    expect(p.homebrewEnv).toEqual({
      HOMEBREW_API_DOMAIN: `${USTC}/api`,
      HOMEBREW_BOTTLE_DOMAIN: USTC,
      HOMEBREW_PIP_INDEX_URL: ALIYUN,
    });
  });

  test("强制 global：官方不通也不切模式", () => {
    const p = decidePlan({ region: "global", cnLocale: true, probes: probes({ [HF_OFFICIAL]: null }) });
    expect(p).toMatchObject({ mode: "global", decidedBy: "setting", modelSource: "huggingface" });
  });

  test("USTC 不通时不设 Homebrew 域名", () => {
    const p = decidePlan({ region: "cn", cnLocale: true, probes: probes({ [USTC]: null }) });
    expect(p.homebrewEnv.HOMEBREW_BOTTLE_DOMAIN).toBeUndefined();
    expect(p.homebrewEnv.HOMEBREW_PIP_INDEX_URL).toBe(ALIYUN);
  });

  test("覆盖地址排最前（去尾斜杠 / GitHub 前缀补尾斜杠）", () => {
    const p = decidePlan({
      region: "global",
      cnLocale: false,
      probes: probes(),
      overrides: { hf: "https://my-hf.example/", pypi: "https://pip.example/simple/", github: "https://gh.example" },
    });
    expect(p.hfEndpoints[0]).toBe("https://my-hf.example");
    expect(p.pypiIndexes[0]).toBe("https://pip.example/simple");
    expect(p.githubPrefixes[0]).toBe("https://gh.example/");
    expect(p.hfEndpoints).toContain(HF_OFFICIAL);
  });

  test("报告失败的主机降级到末尾（官方在 global 下也让位）", () => {
    const p = decidePlan({
      region: "global",
      cnLocale: false,
      probes: probes(),
      failedHosts: new Set(["huggingface.co", "gh-proxy.com"]),
    });
    expect(p.hfEndpoints).toEqual([HF_MIRROR, HF_OFFICIAL]);
    expect(p.githubPrefixes[0]).toBe("");
    expect(p.githubPrefixes[p.githubPrefixes.length - 1]).toBe("https://gh-proxy.com/");
  });
});

describe("子进程环境", () => {
  test("官方直连：不设任何变量（不覆盖用户自己的 pip.conf 等）", () => {
    expect(sourceEnv(decidePlan({ region: "global", cnLocale: false, probes: probes() }))).toEqual({});
  });

  test("国内加速：HF / llama.cpp / pip / uv / brew 全套", () => {
    const env = sourceEnv(decidePlan({ region: "cn", cnLocale: true, probes: probes() }));
    expect(env.HF_ENDPOINT).toBe(HF_MIRROR);
    expect(env.MODEL_ENDPOINT).toBe(`${HF_MIRROR}/`);
    expect(env.PIP_INDEX_URL).toBe(ALIYUN);
    expect(env.UV_DEFAULT_INDEX).toBe(ALIYUN);
    expect(env.UV_PYTHON_INSTALL_MIRROR).toBe(
      "https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download",
    );
    expect(env.HOMEBREW_BOTTLE_DOMAIN).toBe(USTC);
  });
});

describe("GitHub 候选", () => {
  const url = "https://github.com/o/r/releases/download/v1/a.tgz";
  test("国内加速：镜像前缀在前，原始 URL 兜底", () => {
    const c = githubCandidates(url, decidePlan({ region: "cn", cnLocale: true, probes: probes() }));
    expect(c[0]).toBe(`${GITHUB_PREFIX_MIRRORS[0]}${url}`);
    expect(c[c.length - 1]).toBe(url);
    expect(c).toHaveLength(GITHUB_PREFIX_MIRRORS.length + 1);
  });
  test("官方直连：原始 URL 第一", () => {
    expect(githubCandidates(url, decidePlan({ region: "global", cnLocale: false, probes: probes() }))[0]).toBe(url);
  });
});

describe("缓存 / 并发 / 失败上报", () => {
  function setup(opts: { region?: "auto" | "cn" | "global"; lat?: Record<string, Lat>; delayMs?: number } = {}) {
    const calls: string[] = [];
    const settings = { region: opts.region ?? "auto", hf: "", pypi: "", github: "" };
    const lat = opts.lat ?? {};
    const probe: ProbeFn = async (url) => {
      calls.push(url);
      if (opts.delayMs) await Bun.sleep(opts.delayMs);
      const key = Object.keys(lat).find((k) => url.startsWith(k));
      const l = key ? lat[key]! : 100;
      return l === null ? { ok: false, latencyMs: null } : { ok: true, latencyMs: l };
    };
    let now = 1_000_000;
    __setNetSourcesDepsForTest({
      probe,
      settings: () => settings,
      cnLocale: () => false,
      now: () => now,
    });
    return { calls, settings, advance: (ms: number) => (now += ms) };
  }

  test("并发调用共用一次探测，之后命中缓存", async () => {
    const { calls } = setup({ delayMs: 20 });
    const [a, b] = await Promise.all([getSourcePlan(), getSourcePlan()]);
    expect(a).toBe(b);
    const n = calls.length;
    expect(n).toBeGreaterThan(5);
    await getSourcePlan();
    expect(calls.length).toBe(n);
  });

  test("10 分钟过期 / refresh 都会重探", async () => {
    const { calls, advance } = setup();
    await getSourcePlan();
    const n = calls.length;
    advance(9 * 60_000);
    await getSourcePlan();
    expect(calls.length).toBe(n);
    advance(2 * 60_000);
    await getSourcePlan();
    expect(calls.length).toBe(2 * n);
    await getSourcePlan({ refresh: true });
    expect(calls.length).toBe(3 * n);
  });

  test("改了设置立即作废缓存；强制模式仍探测（用于排序）", async () => {
    const { calls, settings } = setup();
    expect((await getSourcePlan()).mode).toBe("global");
    const n = calls.length;
    settings.region = "cn";
    const p = await getSourcePlan();
    expect(p).toMatchObject({ mode: "cn", decidedBy: "setting" });
    expect(calls.length).toBe(2 * n);
  });

  test("探测结果进计划：官方 HF 不通 → cn", async () => {
    setup({ lat: { [HF_OFFICIAL]: null } });
    const p = await getSourcePlan();
    expect(p.mode).toBe("cn");
    expect(p.probes.find((x) => x.url === HF_OFFICIAL)).toMatchObject({ ok: false, latencyMs: null, official: true });
  });

  test("peek：无缓存给地区猜测并在后台探测；有缓存直接返回", async () => {
    const { calls } = setup();
    const guess = peekSourcePlan();
    expect(guess.decidedBy).toBe("locale-guess");
    expect(guess.probes).toEqual([]);
    const real = await getSourcePlan(); // 与 peek 发起的后台探测共用
    expect(peekSourcePlan()).toBe(real);
    expect(real.decidedBy).toBe("probe");
    expect(calls.length).toBeGreaterThan(0);
  });

  test("reportSourceFailure：降级该主机并触发重探，peek 期间仍返回旧计划", async () => {
    const { calls } = setup();
    const before = await getSourcePlan();
    expect(before.hfEndpoints[0]).toBe(HF_OFFICIAL);
    const n = calls.length;
    reportSourceFailure("https://huggingface.co/some/model/resolve/main/x.gguf");
    expect(peekSourcePlan()).toBe(before);
    const after = await getSourcePlan();
    expect(calls.length).toBeGreaterThanOrEqual(2 * n);
    expect(after.hfEndpoints[0]).toBe(HF_MIRROR);
  });
});

describe("与设置表联动（真实 getSetting）", () => {
  test("DOWNLOAD_REGION 强制 cn / 非法值被拒 / 覆盖地址排最前", async () => {
    const { updateSettings } = await import("./db/settings");
    __setNetSourcesDepsForTest({
      probe: async () => ({ ok: true, latencyMs: 50 }),
      cnLocale: () => false,
    });
    try {
      expect((await getSourcePlan()).mode).toBe("global");
      updateSettings({ DOWNLOAD_REGION: "cn", DOWNLOAD_PYPI_INDEX: "https://pip.corp.example/simple" });
      const p = await getSourcePlan();
      expect(p).toMatchObject({ mode: "cn", decidedBy: "setting" });
      expect(p.pypiIndexes[0]).toBe("https://pip.corp.example/simple");
      updateSettings({ DOWNLOAD_REGION: "moon" }); // 非法值：静默拒掉，保持 cn
      expect((await getSourcePlan()).mode).toBe("cn");
    } finally {
      updateSettings({ DOWNLOAD_REGION: "auto", DOWNLOAD_PYPI_INDEX: "" });
    }
  });
});

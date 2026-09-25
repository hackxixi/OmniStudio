/**
 * E3 本机结果汇总：每个量化版本一行 —— JEV 选下一步 / 验收，原生工具调用（A 组），内存与速度。
 *
 * 用法：bun analyze-mac.ts results-mac.jsonl
 */
import { readFileSync } from "node:fs";

type Line = Record<string, unknown> & { tag: string; kind?: string };
const lines = readFileSync(process.argv[2] ?? "results-mac.jsonl", "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l) as Line);

const tags = [...new Set(lines.map((l) => l.tag))];

function auc(scores: { p: number; y: boolean }[]): number {
  const pos = scores.filter((s) => s.y);
  const neg = scores.filter((s) => !s.y);
  let wins = 0;
  for (const a of pos) for (const b of neg) wins += a.p > b.p ? 1 : a.p === b.p ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}

function catchAt10(scores: { p: number; y: boolean }[]): number {
  const passes = scores.filter((s) => s.y).length;
  const fails = scores.length - passes;
  let best = 0;
  for (let t = 0.01; t <= 0.99; t += 0.01) {
    const fa = scores.filter((s) => s.y && s.p < t).length / passes;
    if (fa <= 0.1) best = Math.max(best, scores.filter((s) => !s.y && s.p < t).length / fails);
  }
  return best;
}

const pct = (a: number, b: number) => `${Math.round((a / b) * 100)}%`;
console.log("| 版本 | 权重内存 | 峰值内存 | 预填充 tok/s | 生成 tok/s | JEV 单次 | 选下一步 | 其中工具 | 该反问 | 验收 AUC | 误报≤10% 拦截 | 原生调用通过 | 每题平均 |");
console.log("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
for (const tag of tags) {
  const of = (kind: string) => lines.filter((l) => l.tag === tag && (l.kind ?? "bench") === kind);
  const sel = of("select") as unknown as { correct: boolean; expected: string[] }[];
  const tools = sel.filter((s) => !s.expected.includes("finish") && !s.expected.includes("ask_user"));
  const ask = sel.filter((s) => s.expected.includes("ask_user"));
  const ver = (of("verify") as unknown as { p: number; pass: boolean }[]).map((v) => ({ p: v.p, y: v.pass }));
  const bench = of("bench") as unknown as { pass: boolean; ms: number }[];
  const stats = (of("stats")[0]?.stats ?? {}) as Record<string, number>;
  const gb = (x?: number) => (x === undefined ? "-" : `${x.toFixed(2)} GB`);
  console.log(
    `| ${tag} | ${gb(stats.weights_gb)} | ${gb(stats.peak_gb)} | ${stats.prompt_tps_median ?? "-"} | ${stats.generation_tps_median ?? "-"} | ` +
      `${stats.jev_ms_median ? `${(stats.jev_ms_median / 1000).toFixed(1)} s` : "-"} | ` +
      `${sel.length ? pct(sel.filter((s) => s.correct).length, sel.length) : "-"} | ` +
      `${tools.length ? `${tools.filter((s) => s.correct).length}/${tools.length}` : "-"} | ` +
      `${ask.length ? `${ask.filter((s) => s.correct).length}/${ask.length}` : "-"} | ` +
      `${ver.length ? auc(ver).toFixed(2) : "-"} | ${ver.length ? pct(catchAt10(ver), 1) : "-"} | ` +
      `${bench.length ? pct(bench.filter((b) => b.pass).length, bench.length) : "-"} | ` +
      `${bench.length ? `${Math.round(bench.reduce((a, b) => a + b.ms, 0) / bench.length / 1000)} s` : "-"} |`,
  );
}

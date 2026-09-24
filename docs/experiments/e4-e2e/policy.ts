/**
 * E4 路由策略离线推演：对 e2e.ts 采集的每道题，按策略决定「升级」与否；升级则最终结果取云端 35B，
 * 否则取本地 4B。给出最终准确率、留在本地的比例、云端解码题数与 token、云端 JEV 调用数、平均耗时。
 *
 * 耗时口径：本地总耗时（含本地 JEV 把关）+ 验收耗时（策略用到验收时）+ 升级时云端重做耗时。
 * 过程中信号（领先幅度 / 不一致）触发的升级在真实系统里可以提前打断本地，这里按跑完本地计，是上界。
 *
 * 用法：bun policy.ts results-e4.jsonl
 */
import { readFileSync } from "node:fs";

type Row = {
  task: string;
  local: { pass: boolean; ms: number; jevMs: number };
  steps: { margin: number; agree: boolean }[];
  verify: { p: number; ms: number; inputTokens: number };
  cloud: { pass: boolean; ms: number; promptTokens: number; completionTokens: number };
};

const rows = readFileSync(process.argv[2] ?? "results-e4.jsonl", "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l) as Row);

type Policy = { name: string; usesVerify: boolean; escalate: (r: Row) => boolean };
const minMargin = (r: Row) => Math.min(...r.steps.map((s) => s.margin));
const disagree = (r: Row) => r.steps.some((s) => !s.agree);

const policies: Policy[] = [
  { name: "全本地", usesVerify: false, escalate: () => false },
  { name: "全云端", usesVerify: false, escalate: () => true },
  { name: "领先幅度 < 0.3", usesVerify: false, escalate: (r) => minMargin(r) < 0.3 },
  { name: "本地 JEV 与模型不一致", usesVerify: false, escalate: disagree },
  { name: "不一致 或 幅度 < 0.3", usesVerify: false, escalate: (r) => disagree(r) || minMargin(r) < 0.3 },
  ...[0.5, 0.7, 0.8, 0.9].map((t) => ({ name: `云端验收 < ${t}`, usesVerify: true, escalate: (r: Row) => r.verify.p < t })),
  ...[0.7, 0.8, 0.9].map((t) => ({
    name: `不一致 或 幅度<0.3 或 验收<${t}`,
    usesVerify: true,
    escalate: (r: Row) => disagree(r) || minMargin(r) < 0.3 || r.verify.p < t,
  })),
  { name: "（上界）只在本地真错时升级", usesVerify: false, escalate: (r) => !r.local.pass },
];

console.log(`共 ${rows.length} 题；本地单独通过 ${rows.filter((r) => r.local.pass).length}，云端单独通过 ${rows.filter((r) => r.cloud.pass).length}\n`);
console.log("| 策略 | 最终通过 | 留在本地 | 云端解码题数 | 云端解码 token | 云端 JEV 调用 | 平均耗时 | 升级里本地本来就对的（白花钱） |");
console.log("|---|---|---|---|---|---|---|---|");
for (const p of policies) {
  let pass = 0;
  let up = 0;
  let wasted = 0;
  let tokens = 0;
  let ms = 0;
  for (const r of rows) {
    const e = p.escalate(r);
    const allCloud = p.name === "全云端";
    if (e) {
      up++;
      tokens += r.cloud.promptTokens + r.cloud.completionTokens;
      if (r.local.pass) wasted++;
    }
    pass += (e ? r.cloud.pass : r.local.pass) ? 1 : 0;
    ms += allCloud ? r.cloud.ms : r.local.ms + (p.usesVerify ? r.verify.ms : 0) + (e ? r.cloud.ms : 0);
  }
  const n = rows.length;
  console.log(
    `| ${p.name} | ${Math.round((pass / n) * 100)}% | ${Math.round(((n - up) / n) * 100)}% | ${up} | ${tokens} | ${p.usesVerify ? n : 0} | ` +
      `${(ms / n / 1000).toFixed(1)} s | ${wasted} |`,
  );
}

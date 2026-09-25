/**
 * E6 汇总：按 tag（策略 + 修复组合）统计每题通过次数 / 总次数、整体平均通过率、
 * 平均与中位耗时、平均工具调用次数，输出两张 markdown 表（每组一行 + 任务 × tag）。
 *
 * 只统计，不重新打分：行里的 pass 是当时规则打的分。旧行没有 rep 字段，按 1 次算
 * （新行每轮一行，带 rep=1..N）。
 *
 * 用法：bun summarize.ts results-e6.jsonl
 */
import { readFileSync } from "node:fs";

type Row = {
  tag: string;
  rep?: number;
  task: string;
  kind: string;
  pass: boolean;
  ms: number;
  calls?: { tool: string }[];
};

const file = process.argv[2] ?? "results-e6.jsonl";
const rows: Row[] = readFileSync(file, "utf8")
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l) as Row);

const tags = [...new Set(rows.map((r) => r.tag))];
const taskOrder = [...new Set(rows.map((r) => r.task))];
const median = (xs: number[]) => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
};
const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const f = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`);

// ---------------- 每组一行 ----------------
console.log("| tag | 通过 | 整体通过率 | 耗时平均 | 耗时中位 | 平均工具调用 |");
console.log("|---|---|---|---|---|---|");
for (const tag of tags) {
  const rs = rows.filter((r) => r.tag === tag);
  const passed = rs.filter((r) => r.pass).length;
  console.log(
    `| ${tag} | ${passed}/${rs.length} | ${Math.round((passed / rs.length) * 100)}% | ${f(avg(rs.map((r) => r.ms)))} | ${f(median(rs.map((r) => r.ms)))} | ${avg(rs.map((r) => r.calls?.length ?? 0)).toFixed(1)} |`,
  );
}
const allPassed = rows.filter((r) => r.pass).length;
console.log(
  `| **整体** | **${allPassed}/${rows.length}** | **${Math.round((allPassed / rows.length) * 100)}%** | **${f(avg(rows.map((r) => r.ms)))}** | **${f(median(rows.map((r) => r.ms)))}** | **${avg(rows.map((r) => r.calls?.length ?? 0)).toFixed(1)}** |`,
);

// ---------------- 任务 × tag ----------------
console.log("");
console.log("| 任务 | " + tags.join(" | ") + " |");
console.log("|---|" + tags.map(() => "---|").join(""));
for (const task of taskOrder) {
  const cell = (tag: string) => {
    const rs = rows.filter((r) => r.tag === tag && r.task === task);
    if (!rs.length) return "—";
    return `${rs.filter((r) => r.pass).length}/${rs.length}`;
  };
  console.log(`| ${task} | ${tags.map(cell).join(" | ")} |`);
}

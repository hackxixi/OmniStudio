/**
 * 第二轮结果按任务类别拆开看：近义工具（c-）、不该调工具（l1-）、该反问（ask-）、多步（m-）。
 * 每格是通过数 / 题数；另列「工具选对但参数错」的题数，把调度错误和填参数错误分开。
 *
 * 用法：bun analyze.ts results-r2.jsonl
 */
import { readFileSync } from "node:fs";

type Row = { arm: string; tag?: string; hints?: boolean; task: string; pass: boolean; toolsOk: boolean; argsOk: boolean; answerOk: boolean; ms: number };

const rows = readFileSync(process.argv[2] ?? "results-r2.jsonl", "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l) as Row);

const CATS: [string, RegExp][] = [
  ["近义工具", /^c-/],
  ["不该调", /^l1-/],
  ["该反问", /^ask-/],
  ["多步", /^m-/],
];

const groups = new Map<string, Row[]>();
for (const r of rows) {
  const key = `${r.tag ?? ""} ${r.arm}${r.hints ? "+hints" : ""}`;
  groups.set(key, [...(groups.get(key) ?? []), r]);
}

const cell = (rs: Row[]) => `${rs.filter((r) => r.pass).length}/${rs.length}`;
console.log(`| 组 | 总通过 | ${CATS.map(([n]) => n).join(" | ")} | 选错工具 | 工具对、参数错 | 平均耗时 |`);
console.log(`|---|---|${CATS.map(() => "---|").join("")}---|---|---|`);
for (const [key, rs] of groups) {
  const pct = Math.round((rs.filter((r) => r.pass).length / rs.length) * 100);
  const cats = CATS.map(([, re]) => cell(rs.filter((r) => re.test(r.task))));
  const wrongTool = rs.filter((r) => !r.toolsOk).length;
  const wrongArgs = rs.filter((r) => r.toolsOk && !r.argsOk).length;
  const ms = Math.round(rs.reduce((a, r) => a + r.ms, 0) / rs.length);
  console.log(`| ${key.trim()} | ${pct}% | ${cats.join(" | ")} | ${wrongTool} | ${wrongArgs} | ${ms} ms |`);
}

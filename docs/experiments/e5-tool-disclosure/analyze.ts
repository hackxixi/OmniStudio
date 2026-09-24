/**
 * E5 汇总：每组的通过率（总体 / 按题型）、耗时，C / D 的选组质量（该加载的组选中没有、多加载了没有），
 * 以及模型自己调 load_tools 的情况（E 全靠它，C / D 靠它兜底）。
 *
 * 用法：bun analyze.ts results-e5.jsonl
 */
import { readFileSync } from "node:fs";

type Row = {
  variant: string;
  task: string;
  kind: string;
  needGroups: string[];
  picked: string[];
  loadedByModel: string[];
  loadCalls: number;
  pass: boolean;
  why: string;
  ms: number;
};
const rows = readFileSync(process.argv[2] ?? "results-e5.jsonl", "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l) as Row);

const kinds = ["core", "group", "multi", "none", "ask"];
const kindName: Record<string, string> = { core: "核心", group: "单组", multi: "多步", none: "不该调", ask: "该反问" };
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0;

console.log(`| 组 | 通过 | ${kinds.map((k) => kindName[k]).join(" | ")} | 耗时中位 | 平均 |`);
console.log(`|---|---|${kinds.map(() => "---|").join("")}---|---|`);
for (const v of ["A", "B", "C", "D", "E"]) {
  const rs = rows.filter((r) => r.variant === v);
  if (!rs.length) continue;
  const cell = (k: string) => {
    const x = rs.filter((r) => r.kind === k);
    return `${x.filter((r) => r.pass).length}/${x.length}`;
  };
  console.log(
    `| ${v} | ${rs.filter((r) => r.pass).length}/${rs.length} (${Math.round((rs.filter((r) => r.pass).length / rs.length) * 100)}%) | ` +
      `${kinds.map(cell).join(" | ")} | ${(median(rs.map((r) => r.ms)) / 1000).toFixed(1)} s | ${(rs.reduce((a, r) => a + r.ms, 0) / rs.length / 1000).toFixed(1)} s |`,
  );
}

console.log("\n| 组 | 需要组的题：开局就选中全部所需组 | 不需要组的题：开局多加载了组 | 模型调 load_tools 的题数 | 其中补对了所需组 |");
console.log("|---|---|---|---|---|");
for (const v of ["C", "D", "E"]) {
  const rs = rows.filter((r) => r.variant === v);
  if (!rs.length) continue;
  const need = rs.filter((r) => r.needGroups.length);
  const covered = need.filter((r) => r.needGroups.every((g) => r.picked.includes(g))).length;
  const noNeed = rs.filter((r) => !r.needGroups.length);
  const extra = noNeed.filter((r) => r.picked.length).length;
  const loaders = rs.filter((r) => r.loadCalls > 0);
  const fixed = loaders.filter((r) => r.needGroups.every((g) => r.picked.includes(g) || r.loadedByModel.includes(g)) && r.needGroups.length).length;
  console.log(`| ${v} | ${covered}/${need.length} | ${extra}/${noNeed.length} | ${loaders.length} | ${fixed} |`);
}

console.log("\n失败明细：");
for (const r of rows.filter((x) => !x.pass)) console.log(`${r.variant} ${r.task.padEnd(20)} picked=[${r.picked}] +load=[${r.loadedByModel}] ${r.why}`);

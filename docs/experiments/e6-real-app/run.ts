/**
 * E6：在真实 App 里用本地候选模型跑 E5 的任务做回归。每题一次 `omi agent run --json`（真实 Agent、真实工具），
 * 从事件流里取工具调用与最终回答，用 E5 的规则打分。
 *
 * 真实 App 里没有 E5 的感知 / 日程工具（转写、OCR、提醒、日历、联系人、邮件），用到它们的 7 题跳过；
 * 联网 / 生成 / 知识库的结果取决于外部服务或未预置的数据，这些题只检查「调了哪个工具、参数对不对」，
 * 不检查回答内容。笔记、记忆、工作区文件、生图记录都由 seed.ts 预置，这些题连回答一起检查。
 *
 * 用法：OMNI_DATA_DIR=<目录> WORKSPACE=<目录> TAG=routed bun run.ts [任务 id 前缀]
 */
import { spawnSync } from "node:child_process";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { TASKS, score, type Call, type Task } from "../e5-tool-disclosure/tasks";
import { READ_ONLY } from "../e5-tool-disclosure/tools";

const WORKSPACE = process.env.WORKSPACE!;
const TAG = process.env.TAG ?? "?";
const OUT = process.env.OUT ?? "results-e6.jsonl";
const OMI = path.resolve(import.meta.dir, "../../../apps/studio/bin/omi.ts");

/** 真实 App 里不存在的能力。 */
const UNSUPPORTED = /^(transcribe_audio|ocr_document|set_reminder|create_calendar_event|contacts_lookup|send_email)$/;
const unsupported = (t: Task) => t.steps.some((s) => Object.keys(s).every((k) => UNSUPPORTED.test(k)));
/**
 * 真实 App 会把相关记忆自动拼进用户消息（笔记保存时也会写一条索引记忆），这两题不调工具、直接答对也算对。
 */
const ANSWER_IS_ENOUGH = new Set(["memory-recall", "note-lookup"]);
/** 数据已预置、回答可以核对的题；其余只核对工具与参数。 */
const CHECK_ANSWER = new Set(["note-lookup", "memory-recall", "file-read", "file-find", "l1-storm", "l1-poem", "l1-divide", "l1-translate"]);

const FILES: Record<string, string> = {
  "notes/todo.md": "- [x] 提交周报\n- [ ] 报销差旅发票\n- [ ] 预约牙医\n- [x] 续费域名\n",
  "docs/plan.md": "# 上线计划\n\n- 功能冻结：9 月 30 日\n- 上线日期：10 月 8 日\n- 负责人：张伟\n",
  "finance/budget-2026.xlsx": "placeholder spreadsheet",
};
function resetWorkspace() {
  for (const [rel, content] of Object.entries(FILES)) {
    mkdirSync(path.dirname(path.join(WORKSPACE, rel)), { recursive: true });
    writeFileSync(path.join(WORKSPACE, rel), content);
  }
}

type Line = { type: string; event?: { kind: string; toolName?: string; args?: string | null; output?: string | null }; text?: string; ok?: boolean; error?: string };

const only = process.argv[2];
const tasks = TASKS.filter((t) => !unsupported(t) && (!only || t.id.startsWith(only)));
let pass = 0;
let ms = 0;
for (const task of tasks) {
  resetWorkspace();
  const t0 = performance.now();
  const r = spawnSync("bun", [OMI, "agent", "run", task.request, "--json", "--workspace", WORKSPACE, "--timeout", "600000"], {
    encoding: "utf8",
    env: process.env,
    maxBuffer: 64 * 1024 * 1024,
  });
  const elapsed = Math.round(performance.now() - t0);
  const lines = r.stdout
    .split("\n")
    .filter(Boolean)
    .flatMap((l) => {
      try {
        return [JSON.parse(l) as Line];
      } catch {
        return [];
      }
    });
  const calls: Call[] = [];
  let routing = "";
  for (const l of lines) {
    const e = l.event;
    if (l.type !== "event" || !e) continue;
    if (e.kind === "tool_start" && e.toolName) {
      let args: Record<string, unknown> = {};
      try {
        args = e.args ? (JSON.parse(e.args) as Record<string, unknown>) : {};
      } catch {
        args = { _raw: e.args };
      }
      calls.push({ tool: e.toolName, args, result: "" });
    }
    if (e.kind === "status" && e.toolName === "tool_routing") routing = e.output ?? "";
  }
  const result = lines.find((l) => l.type === "result");
  const answer = result?.text ?? "";
  const answerOk = task.answer.every((re) => re.test(answer));
  const scored =
    ANSWER_IS_ENOUGH.has(task.id) && answerOk && !calls.some((c) => !READ_ONLY.has(c.tool))
      ? { pass: true, why: "" }
      : score(CHECK_ANSWER.has(task.id) ? task : { ...task, answer: [] }, calls, answer, READ_ONLY);
  const ok = scored.pass && result?.ok !== false;
  pass += ok ? 1 : 0;
  ms += elapsed;
  appendFileSync(
    OUT,
    `${JSON.stringify({ tag: TAG, task: task.id, kind: task.kind, pass: ok, why: result?.ok === false ? `run failed: ${result.error}` : scored.why, calls, routing, answer: answer.slice(0, 400), ms: elapsed })}\n`,
  );
  console.log(`${TAG} ${task.id.padEnd(20)} ${ok ? "PASS" : "fail"} ${String(elapsed).padStart(6)}ms [${calls.map((c) => c.tool).join(" → ")}] ${routing ? `(${routing}) ` : ""}${ok ? "" : scored.why || result?.error || ""}`);
}
console.log(`== ${TAG}: ${pass}/${tasks.length} (${Math.round((pass / tasks.length) * 100)}%)  avg ${(ms / tasks.length / 1000).toFixed(1)}s`);

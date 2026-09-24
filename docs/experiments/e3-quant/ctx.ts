/**
 * 短任务到底吃多少上下文：拿 E1 第二轮 35B 的真实轨迹，按 bench.ts 的原样拼出请求，
 * 让 SGLang 只做预填充（max_tokens=1）读回 usage.prompt_tokens；JEV 请求读 OpenJev 回报的 usage。
 *
 * 用法（ycs2）：bun ctx.ts
 */
import { readFileSync } from "node:fs";
import { CALENDAR, NOW, TASKS, TOOLS } from "../e1-jev-orchestration/tasks2";

const LLM = process.env.MID_URL ?? "http://127.0.0.1:30000";
const MODEL = process.env.MID_MODEL ?? "Qwen/Qwen3.6-35B-A3B";
const JEV_URL = process.env.JEV_URL ?? "http://127.0.0.1:18110";
const JEV_KEY = readFileSync(`${process.env.HOME}/jev/openjev.key`, "utf8").trim();

type Call = { tool: string; args: Record<string, unknown>; result: string };
type Row = { arm: string; tag?: string; hints?: boolean; task: string; calls: Call[]; answer: string };
const rows = readFileSync("../e1-jev-orchestration/results-r2.jsonl", "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l) as Row)
  .filter((r) => r.tag === "35B" && r.arm === "C" && !r.hints);

const tools = TOOLS.map((t) => ({ type: "function", function: { name: t.name, description: `${t.description} ${t.hint ?? ""}`, parameters: t.parameters } }));

async function prefill(messages: unknown[], withTools: boolean): Promise<number> {
  const r = await fetch(`${LLM}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, messages, max_tokens: 1, temperature: 0, chat_template_kwargs: { enable_thinking: false }, ...(withTools ? { tools } : {}) }),
  });
  const j = (await r.json()) as { usage: { prompt_tokens: number } };
  return j.usage.prompt_tokens;
}

async function jevTokens(content: string, criteria: Record<string, string>): Promise<number> {
  const r = await fetch(`${JEV_URL}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${JEV_KEY}` },
    body: JSON.stringify({
      model: "jev-latest",
      state: { messages: [{ role: "user", content }] },
      questions: { next: { type: "choice", instructions: "Pick the next action.", criteria } },
    }),
  });
  const j = (await r.json()) as { usage?: Record<string, number> };
  return j.usage?.input_tokens ?? j.usage?.prompt_tokens ?? Number.NaN;
}

const native: number[] = [];
const jevs: number[] = [];
const noTools: number[] = [];
for (const row of rows) {
  const task = TASKS.find((t) => t.id === row.task)!;
  const files = task.attachments?.map((a) => a.name).join(", ");
  const system =
    `You are a helpful assistant with tools. Current local time: ${NOW}. ${CALENDAR}` +
    (files ? ` The user attached these files: ${files}.` : " The user attached no files.");
  // 最后一轮（所有工具结果都已回填）就是这一题的上下文峰值
  const messages: unknown[] = [
    { role: "system", content: system },
    { role: "user", content: task.request },
  ];
  for (const c of row.calls) {
    messages.push({ role: "assistant", content: `<tool_call>\n<function=${c.tool}>\n${Object.entries(c.args).map(([k, v]) => `<parameter=${k}>\n${v}\n</parameter>`).join("\n")}\n</function>\n</tool_call>` });
    messages.push({ role: "tool", content: c.result });
  }
  native.push(await prefill(messages, true));
  noTools.push(await prefill(messages, false));
  const history = row.calls.map((c, i) => `${i + 1}. ${c.tool}(${JSON.stringify(c.args)}) → ${c.result}`).join("\n") || "(none yet)";
  const criteria: Record<string, string> = { finish: "Reply to the user now.", ask_user: "Ask the user a question first." };
  for (const t of TOOLS) criteria[t.name] = `Call ${t.name} next: ${t.description}`;
  jevs.push(await jevTokens(`Current local time: ${NOW}. ${CALENDAR}\nUser request: ${task.request}\n\nActions taken so far:\n${history}`, criteria));
}

const stat = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return `min ${s[0]}  中位 ${s[Math.floor(s.length / 2)]}  最大 ${s[s.length - 1]}`;
};
console.log(`原生工具调用（23 个工具 + 历史）：${stat(native)}`);
console.log(`其中不带工具定义：${stat(noTools)}  → 工具定义约占 ${Math.round(native[0]! - noTools[0]!)} token`);
console.log(`JEV 选下一步（25 个选项）：${stat(jevs)}`);

/**
 * E5：工具精简与渐进式披露。本地 Qwen3.5-4B（MLX，mlx_jev_server --prefix-cache）用真实 Agent 的工具定义做 33 题。
 *
 *   A  现状全量：真实 34 个工具 + 感知 2 + 日程通讯 4 = 40 个，一次全给
 *   B  精简扁平：核心 9 个（知识 / 笔记 / 记忆合成 recall，文件查找合成 find，去掉流程控制与 JEV）+ 全部工具组，共 22 个
 *   C  核心 + JEV 选组：开始时同一个 4B 当 JEV 在工具组里选（可多选一组），其余组可用 load_tools 补
 *   D  核心 + 向量检索选组：bge-small 向量相似度选组，其余同 C
 *   E  核心 + 只靠模型自己 load_tools
 *
 * 用法（Mac）：LOCAL_URL=http://127.0.0.1:18130 EMBED_URL=http://127.0.0.1:18132 bun bench.ts A,B,C,D,E [任务 id 前缀]
 */
import { appendFileSync } from "node:fs";
import { CALENDAR, NOW, TASKS, score, type Call, type Task } from "./tasks";
import { GROUPS, READ_ONLY, resetStubs, runStub, variantTools, type ToolDef, type Variant } from "./tools";

const LOCAL_URL = process.env.LOCAL_URL ?? "http://127.0.0.1:18130";
const EMBED_URL = process.env.EMBED_URL ?? "http://127.0.0.1:18132";
const OUT = process.env.OUT ?? "results-e5.jsonl";
const MAX_STEPS = 6;
/** C 组：第二名的概率不低于这个值时，把第二组也一起加载。 */
const SECOND_GROUP_P = 0.25;

type Msg = { role: string; content: string };

async function chat(messages: Msg[], tools: ToolDef[]) {
  const r = await fetch(`${LOCAL_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages,
      tools: tools.map((t) => ({ type: "function", function: t })),
      temperature: 0,
      max_tokens: 1024,
      chat_template_kwargs: { enable_thinking: false },
    }),
  });
  if (!r.ok) throw new Error(`chat HTTP ${r.status}`);
  const j = (await r.json()) as { choices: { message: { content: string } }[] };
  return j.choices[0]?.message.content ?? "";
}

/** qwen3_coder XML（数组 / 对象参数是 JSON 文本，解析回来）与 hermes JSON 两种写法。 */
function parseToolCalls(text: string): { name: string; args: Record<string, unknown> }[] {
  const out: { name: string; args: Record<string, unknown> }[] = [];
  for (const block of text.matchAll(/<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g)) {
    const body = block[1] ?? "";
    const fn = body.match(/<function=([^>\s]+)>([\s\S]*?)(?:<\/function>|$)/);
    if (fn) {
      const args: Record<string, unknown> = {};
      for (const p of fn[2]!.matchAll(/<parameter=([^>\s]+)>\n?([\s\S]*?)\n?<\/parameter>/g)) {
        const raw = p[2]!.trim();
        let v: unknown = raw;
        if (/^[[{]/.test(raw)) {
          try {
            v = JSON.parse(raw);
          } catch {
            // 保留原文
          }
        }
        args[p[1]!] = v;
      }
      out.push({ name: fn[1]!, args });
      continue;
    }
    try {
      const j = JSON.parse(body.trim()) as { name: string; arguments?: Record<string, unknown> };
      out.push({ name: j.name, args: j.arguments ?? {} });
    } catch {
      // 写坏的调用算没调
    }
  }
  return out;
}
const strip = (t: string) => t.replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();

function systemPrompt(task: Task, v: Variant, loaded: string[]): string {
  const files = task.attachments?.join(", ");
  const base =
    `You are a helpful assistant with tools. Current local time: ${NOW}. ${CALENDAR}` +
    (files ? ` The user attached these files: ${files}.` : " The user attached no files.") +
    " Use a tool only when it is needed; when you have what you need, answer the user directly in their language." +
    " If information you need from the user is missing, ask them instead of guessing." +
    " When you produce an image, audio or video, tell the user its reference (e.g. img_1 / aud_1 / vid_1).";
  if (v === "A" || v === "B") return base;
  const rest = Object.entries(GROUPS).filter(([g]) => !loaded.includes(g));
  return rest.length
    ? `${base} More tools exist in groups that are not loaded yet — ${rest.map(([, g]) => g.summary).join("; ")}. ` +
        "If your current tools cannot do what the user asked, call load_tools with the group name first."
    : base;
}

// ---------------------------------------------------------------------------
// 选组
// ---------------------------------------------------------------------------

/** 第一名（不是 none 时）加载；第二名满足条件且不是 none 也加载。 */
function topGroups(scores: Record<string, number>, secondToo: (second: [string, number]) => boolean): string[] {
  const [first, second] = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const out: string[] = [];
  if (first && first[0] !== "none") out.push(first[0]);
  if (second && second[0] !== "none" && secondToo(second)) out.push(second[0]);
  return out;
}

async function pickByJev(task: Task): Promise<{ groups: string[]; probs: Record<string, number> }> {
  const criteria: Record<string, string> = {
    none: "None of these: the request only needs web search, reading or writing files, the user's knowledge base / notes / memory, or no tool at all.",
  };
  for (const [g, def] of Object.entries(GROUPS)) criteria[g] = `The request needs the ${def.summary} tools.`;
  const files = task.attachments?.join(", ") ?? "none";
  const r = await fetch(`${LOCAL_URL}/v1/systemone`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: "jev-latest",
      state: { messages: [{ role: "user", content: `Attached files: ${files}\nUser request: ${task.request}` }] },
      questions: {
        group: {
          type: "choice",
          instructions: "Which group of tools does the assistant need to handle this request? Pick the group needed first.",
          criteria,
        },
      },
    }),
  });
  const j = (await r.json()) as { answers: { group: { probabilities: Record<string, number> } } };
  const probs = j.answers.group.probabilities;
  return { groups: topGroups(probs, (second) => second[1] >= SECOND_GROUP_P), probs };
}

const groupTexts: Record<string, string> = {
  none: "search the web, open a web page, read or write or edit files, find files, search the company knowledge base, my notes, remember facts about me, answer questions, translate, explain",
  ...Object.fromEntries(Object.entries(GROUPS).map(([g, d]) => [g, `${d.summary}. ${d.tools.map((t) => t.description.split(".")[0]).join(". ")}`])),
};
let groupVecs: Record<string, number[]> | null = null;

async function embed(texts: string[]): Promise<number[][]> {
  const r = await fetch(`${EMBED_URL}/v1/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ input: texts }),
  });
  const j = (await r.json()) as { data: { embedding: number[] }[] };
  return j.data.map((d) => d.embedding);
}
const cos = (a: number[], b: number[]) => {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i]! * b[i]!;
    na += a[i]! ** 2;
    nb += b[i]! ** 2;
  }
  return d / Math.sqrt(na * nb);
};

async function pickByEmbedding(task: Task): Promise<{ groups: string[]; probs: Record<string, number> }> {
  if (!groupVecs) {
    const keys = Object.keys(groupTexts);
    const vecs = await embed(keys.map((k) => groupTexts[k]!));
    groupVecs = Object.fromEntries(keys.map((k, i) => [k, vecs[i]!]));
  }
  const [q] = await embed([`${task.request}${task.attachments ? ` (attached: ${task.attachments.join(", ")})` : ""}`]);
  const sims = Object.fromEntries(Object.entries(groupVecs).map(([k, v]) => [k, cos(q!, v)]));
  const best = Math.max(...Object.values(sims));
  return { groups: topGroups(sims, (second) => second[1] >= best - 0.02), probs: sims };
}

// ---------------------------------------------------------------------------

async function run(task: Task, v: Variant) {
  resetStubs();
  const t0 = performance.now();
  let pick: { groups: string[]; probs: Record<string, number> } = { groups: [], probs: {} };
  if (v === "C") pick = await pickByJev(task);
  if (v === "D") pick = await pickByEmbedding(task);
  const pickMs = Math.round(performance.now() - t0);
  const loaded = [...pick.groups];
  const tools = variantTools(v, loaded);
  const initialTools = tools.map((t) => t.name);
  const messages: Msg[] = [
    { role: "system", content: systemPrompt(task, v, loaded) },
    { role: "user", content: task.request },
  ];
  const calls: Call[] = [];
  let answer = "";
  let loadCalls = 0;
  for (let step = 0; step < MAX_STEPS; step++) {
    const content = await chat(messages, tools);
    const parsed = parseToolCalls(content);
    if (!parsed.length) {
      answer = strip(content);
      break;
    }
    messages.push({ role: "assistant", content });
    let asked = false;
    for (const c of parsed) {
      const result = runStub(c.name, c.args);
      calls.push({ tool: c.name, args: c.args, result });
      messages.push({ role: "tool", content: result });
      if (c.name === "load_tools") {
        loadCalls++;
        const g = String(c.args.group ?? "");
        if (GROUPS[g] && !loaded.includes(g)) loaded.push(g);
      }
      if (c.name === "ask_user") asked = true;
    }
    if (asked) break;
  }
  const s = score(task, calls, answer, READ_ONLY);
  const row = {
    variant: v,
    task: task.id,
    kind: task.kind,
    needGroups: task.groups,
    picked: pick.groups,
    pickProbs: pick.probs,
    loadedByModel: loaded.filter((g) => !pick.groups.includes(g)),
    loadCalls,
    initialTools,
    calls: calls.map((c) => ({ tool: c.tool, args: c.args })),
    answer: answer.slice(0, 400),
    pass: s.pass,
    why: s.why,
    pickMs,
    ms: Math.round(performance.now() - t0),
  };
  appendFileSync(OUT, `${JSON.stringify(row)}\n`);
  return row;
}

const variants = (process.argv[2] ?? "A,B,C,D,E").split(",") as Variant[];
const only = process.argv[3];
for (const v of variants) {
  let pass = 0;
  let ms = 0;
  const tasks = TASKS.filter((t) => !only || t.id.startsWith(only));
  for (const task of tasks) {
    const r = await run(task, v);
    pass += r.pass ? 1 : 0;
    ms += r.ms;
    console.log(
      `${v} ${task.id.padEnd(20)} ${r.pass ? "PASS" : "fail"} ${String(r.ms).padStart(6)}ms  picked=[${r.picked.join(",")}] +load=[${r.loadedByModel.join(",")}]  ` +
        `[${r.calls.map((c) => c.tool).join(" → ")}] ${r.why}`,
    );
  }
  console.log(`== ${v}: ${pass}/${tasks.length} (${Math.round((pass / tasks.length) * 100)}%)  avg ${(ms / tasks.length / 1000).toFixed(1)}s\n`);
}

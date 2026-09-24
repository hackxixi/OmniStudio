/**
 * E2：多小的 JEV 还能用？同一套 OpenJev 分别架在 Qwen3.5-0.8B / 2B / 4B / 9B 与 Qwen3.6-35B-A3B 上，
 * 三项考试全部用 E1 第二轮的任务与数据，逐项给出准确率与校准：
 *
 *   select  选下一步：以 35B 原生（第二轮全对）的轨迹为标准答案做「教师强制」——每个决策点的历史
 *           都是正确的前几步，JEV 只需选出下一步（工具 / finish / ask_user）。与 E1 的 B 组同一问法。
 *   verify  验收：对第二轮 527 条运行（有真值）用 v2 问法打「处理对了没有」，算 AUC 与误报约束下的拦截率。
 *   route   自知之明：在请求刚到时问「本地小模型能独立做对吗」，与该尺寸小模型（A 组、带 HINTS）的
 *           真实成败对照算 AUC —— 这是「判断做不了就直接交给云端」的依据。
 *
 * 用法（ycs2）：JEV_URL=http://127.0.0.1:18120 TAG=4B bun sweep.ts [select,verify,route]
 */
import { appendFileSync, readFileSync } from "node:fs";
import { CALENDAR, NOW, TASKS, TOOLS } from "../e1-jev-orchestration/tasks2";

const JEV_URL = process.env.JEV_URL ?? "http://127.0.0.1:18120";
const JEV_KEY = readFileSync(process.env.JEV_KEY_FILE ?? `${process.env.HOME}/jev/openjev.key`, "utf8").trim();
const TAG = process.env.TAG ?? "?";
const R2 = process.env.R2 ?? "../e1-jev-orchestration/results-r2.jsonl";
const OUT = process.env.OUT ?? "results-e2.jsonl";

type Call = { tool: string; args: Record<string, unknown>; result: string };
type Row = { arm: string; tag?: string; hints?: boolean; task: string; pass: boolean; answer: string; calls: Call[] };

const rows = readFileSync(R2, "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l) as Row);

async function jev(state: unknown, questions: Record<string, unknown>) {
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(`${JEV_URL}/v1/systemone`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${JEV_KEY}` },
      body: JSON.stringify({ state, model: "jev-latest", questions }),
    });
    const j = (await r.json().catch(() => ({}))) as { answers?: Record<string, Record<string, unknown>> };
    if (r.ok) return j.answers ?? {};
    if (attempt >= 2) throw new Error(`jev HTTP ${r.status} ${JSON.stringify(j).slice(0, 200)}`);
    await Bun.sleep(1000);
  }
}

const taskById = new Map(TASKS.map((t) => [t.id, t]));

/** 与 E1 bench.ts（HINTS=1）一致的上下文。 */
function context(taskId: string, calls: Call[]): string {
  const task = taskById.get(taskId)!;
  const files = task.attachments?.map((a) => `${a.name} (${a.kind})`).join(", ") ?? "none";
  const history = calls.length
    ? calls.map((c, i) => `${i + 1}. ${c.tool}(${JSON.stringify(c.args)}) → ${c.result}`).join("\n")
    : "(none yet)";
  return `Current local time: ${NOW}. ${CALENDAR}\nAttached files: ${files}\nUser request: ${task.request}\n\nActions taken so far:\n${history}`;
}

function auc(scores: { p: number; y: boolean }[]): number {
  const pos = scores.filter((s) => s.y);
  const neg = scores.filter((s) => !s.y);
  if (!pos.length || !neg.length) return Number.NaN;
  let wins = 0;
  for (const a of pos) for (const b of neg) wins += a.p > b.p ? 1 : a.p === b.p ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}

const record = (kind: string, data: Record<string, unknown>) => appendFileSync(OUT, `${JSON.stringify({ tag: TAG, kind, ...data })}\n`);

// ---------------------------------------------------------------------------
// select
// ---------------------------------------------------------------------------

const FINISH =
  "Reply to the user now. Nothing else needs to be looked up, transcribed, read, generated, saved or scheduled: " +
  "either the actions above already did everything the request asks for, or the request can be answered directly " +
  "without any tool (translation, rewriting, arithmetic, general explanation).";
const ASK_USER =
  "Ask the user a question first: the request is missing information that only the user can provide " +
  "(what exactly, when, to whom, how much), so no tool can be called correctly yet.";

async function select() {
  const gold = rows.filter((r) => r.tag === "35B" && r.arm === "C" && !r.hints && r.pass);
  let n = 0;
  let ok = 0;
  const byKind: Record<string, [number, number]> = {};
  const cal: { margin: number; correct: boolean }[] = [];
  for (const g of gold) {
    const task = taskById.get(g.task)!;
    for (let k = 0; k <= g.calls.length; k++) {
      const history = g.calls.slice(0, k);
      const expected = new Set<string>();
      if (task.ask) expected.add("ask_user");
      else if (k === g.calls.length) expected.add("finish");
      else if (task.anyOrder) {
        const done = history.map((c) => c.tool);
        for (const s of task.steps) if (!done.includes(s.tool)) expected.add(s.tool);
      } else expected.add(g.calls[k]!.tool);
      const criteria: Record<string, string> = { finish: FINISH };
      if (k === 0) criteria.ask_user = ASK_USER;
      for (const t of TOOLS) criteria[t.name] = `Call ${t.name} next: ${t.description}`;
      const a = (
        await jev(
          { messages: [{ role: "user", content: context(g.task, history) }] },
          {
            next: {
              type: "choice",
              instructions:
                "You are deciding the assistant's next action for the user request above. Pick the tool that must be " +
                "called next to make progress, or 'finish' if no further tool call is needed. Do not repeat an action " +
                "that already succeeded.",
              criteria,
            },
          },
        )
      ).next as { choice: string; probabilities: Record<string, number> };
      const probs = Object.values(a.probabilities).sort((x, y) => y - x);
      const margin = (probs[0] ?? 0) - (probs[1] ?? 0);
      const correct = expected.has(a.choice);
      const kind = task.ask ? "ask" : k === g.calls.length ? "finish" : "tool";
      byKind[kind] = [(byKind[kind]?.[0] ?? 0) + (correct ? 1 : 0), (byKind[kind]?.[1] ?? 0) + 1];
      n++;
      if (correct) ok++;
      cal.push({ margin, correct });
      record("select", { task: g.task, step: k, expected: [...expected], choice: a.choice, margin, correct });
      if (task.ask) break;
    }
  }
  const confident = cal.filter((c) => c.margin >= 0.3);
  const unsure = cal.filter((c) => c.margin < 0.3);
  const rate = (xs: { correct: boolean }[]) => (xs.length ? `${Math.round((xs.filter((x) => x.correct).length / xs.length) * 100)}%` : "-");
  console.log(
    `[${TAG}] select: ${ok}/${n} (${Math.round((ok / n) * 100)}%)  ` +
      Object.entries(byKind)
        .map(([k, [a, b]]) => `${k} ${a}/${b}`)
        .join("  ") +
      `  | margin≥0.3: ${rate(confident)} of ${confident.length}, <0.3: ${rate(unsure)} of ${unsure.length}`,
  );
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

const VERIFY_V2 =
  "Did the assistant handle this request correctly? Correct means one of two things. (1) The request was complete: the " +
  "assistant took exactly the actions it needed — no unnecessary tools, every detail (names, dates, times, sizes, amounts, " +
  "recipients) matching what the user asked and what earlier results said — and the reply truthfully reports what was done. " +
  "(2) The request was missing information only the user can give: the assistant took no action and asked the user for it.";

/** 在误报率 ≤ maxFa 的前提下能拦下多少失败（「过线」= p 低于阈值 → 判为没做对）。 */
function catchAtFa(scores: { p: number; y: boolean }[], maxFa: number) {
  let best = { t: 0, caught: 0, fa: 0 };
  const passes = scores.filter((s) => s.y).length;
  const fails = scores.length - passes;
  for (let t = 0.05; t <= 0.99; t += 0.01) {
    const fa = scores.filter((s) => s.y && s.p < t).length / passes;
    const caught = scores.filter((s) => !s.y && s.p < t).length / fails;
    if (fa <= maxFa && caught > best.caught) best = { t, caught, fa };
  }
  return best;
}

async function verify() {
  const scores: { p: number; y: boolean }[] = [];
  for (const r of rows) {
    const ans = await jev(
      {
        messages: [
          { role: "user", content: context(r.task, r.calls).replace("Actions taken so far:", "Actions taken:") },
          { role: "assistant", content: r.answer },
        ],
      },
      { done: { type: "noul", instructions: VERIFY_V2 } },
    );
    const p = (ans.done as { noul: number }).noul;
    scores.push({ p, y: r.pass });
    record("verify", { run: `${r.tag}${r.arm}${r.hints ? "+h" : ""}`, task: r.task, pass: r.pass, p });
  }
  const pct = (x: number) => `${Math.round(x * 100)}%`;
  const at5 = catchAtFa(scores, 0.05);
  const at10 = catchAtFa(scores, 0.1);
  console.log(
    `[${TAG}] verify: AUC ${auc(scores).toFixed(3)}  拦截@误报≤5% ${pct(at5.caught)} (t=${at5.t.toFixed(2)})  ` +
      `@误报≤10% ${pct(at10.caught)} (t=${at10.t.toFixed(2)})`,
  );
}

// ---------------------------------------------------------------------------
// route
// ---------------------------------------------------------------------------

const ROUTE_Q =
  "The assistant answering this request is a very small on-device language model. Can it handle the request correctly on " +
  "its own? Small models often fail at: choosing between similar tools, date / time arithmetic, currency math, carrying " +
  "exact details from one step into the next, plans with three or more steps, and noticing when information is missing. " +
  "Answer false if it is likely to make any mistake.";

async function route() {
  for (const size of ["0.8B", "2B", "4B"]) {
    const labels = rows.filter((r) => r.tag === size && r.arm === "A" && r.hints);
    if (!labels.length) continue;
    const scores: { p: number; y: boolean }[] = [];
    for (const r of labels) {
      const ans = await jev(
        { messages: [{ role: "user", content: context(r.task, []) }] },
        { local_ok: { type: "noul", instructions: ROUTE_Q } },
      );
      const p = (ans.local_ok as { noul: number }).noul;
      scores.push({ p, y: r.pass });
      record("route", { target: size, task: r.task, pass: r.pass, p });
    }
    // 按 p 从低到高把请求交给云端：交出去一半时，本地留下的那一半里有多少是做对的
    const sorted = [...scores].sort((a, b) => b.p - a.p);
    const keep = sorted.slice(0, Math.ceil(sorted.length / 2));
    const base = scores.filter((s) => s.y).length / scores.length;
    console.log(
      `[${TAG}] route→${size}: AUC ${auc(scores).toFixed(3)}  本地基线 ${Math.round(base * 100)}%  ` +
        `留下最有把握的一半后本地正确率 ${Math.round((keep.filter((s) => s.y).length / keep.length) * 100)}%`,
    );
  }
}

const parts = (process.argv[2] ?? "select,verify,route").split(",");
if (parts.includes("select")) await select();
if (parts.includes("route")) await route();
if (parts.includes("verify")) await verify();

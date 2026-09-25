/**
 * 精简路由策略的「云端验收 + 升级」（设置 `AGENT_VERIFY_MODE`）。
 *
 * 本地小模型做完一轮后，把「请求 + 这一轮的动作与结果 + 回复」交给**云端** JEV 判一次「处理对了没有」：
 * - `report`：只把结论写进轨迹；
 * - `escalate`：低于阈值时，换成设置里指定的云端模型接着这一轮往下做（补遗漏、改错），不从头重做。
 *
 * 依据 E4（docs/experiments/e4-e2e）：本地 4B 做题 + 云端 35B JEV 验收（p < 0.9 升级）31 题全对，
 * 84% 留在本地，云端解码 token 比全云端少 81%；验收问法用 v2（「做对了，或者该问的问了」），
 * 否则会把「信息不足时正确反问」判成失败。验收必须走云端：E2 里本地 4B 当验收员只拦得下 24% 的失败。
 */
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { SYSTEMONE_DEFAULT_MODEL } from "../shared/systemone";
import { isEmptyResult, isUnansweredAsk, SEARCH_TOOLS } from "./agent-loop-guard";
import { runSystemOne } from "./systemone";

export type VerifyMode = "off" | "report" | "escalate";

export function parseVerifyMode(value: unknown): VerifyMode {
  return value === "report" || value === "escalate" ? value : "off";
}

/**
 * 默认阈值 0.7。E4 桩环境用 0.9 只升级 16%，但 E6 真实 App 里 0.9 升级了 56%，
 * 多出来的主要是 0.73～0.89 的边界判定（回答本身没错）；0.7 时降到 28%。
 */
export const DEFAULT_VERIFY_THRESHOLD = 0.7;

/** 阈值：设置值 → 0～1 之间的数，非法值回落默认值。 */
export function parseVerifyThreshold(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 && n < 1 ? n : DEFAULT_VERIFY_THRESHOLD;
}

/** E4 验证过的 v2 问法（英文：JEV 骨干的判定说明是英文训练的，照原样用）。 */
export const VERIFY_QUESTION =
  "Did the assistant handle this request correctly? Correct means one of two things. (1) The request was complete: the " +
  "assistant took exactly the actions it needed — no unnecessary tools, every detail (names, dates, times, sizes, amounts, " +
  "recipients) matching what the user asked and what earlier results said — and the reply truthfully reports what was done. " +
  "(2) The request was missing information only the user can give: the assistant took no action and asked the user for it.";

const RESULT_CHARS = 400;
const REPLY_CHARS = 2000;

type Block = { type: string; text?: string; name?: string; arguments?: unknown };
type TurnMessage = { role?: string; content?: unknown; toolName?: string; isError?: boolean; details?: { error?: unknown } };

/**
 * 没人应答的 `ask_user` 在验收里写成「已经问了、等用户回答」。无头运行里它的原始结果是一条报错
 * （「不可用」），JEV 会把这条失败当成没做对 —— E6 里「给他发邮件」「帮我画张图」因此判 0.18 / 0.27，
 * 可这两题正确做法恰恰就是反问。
 */
const ASKED_AND_WAITING = "(question shown to the user; waiting for their answer)";

/** 工具结果正文：去掉循环守卫追加的「【提示】」块（那是给模型看的，不是工具的结果）。 */
function resultTextOf(content: unknown): string {
  if (!Array.isArray(content)) return textOf(content);
  return textOf((content as Block[]).filter((b) => !(b.type === "text" && b.text?.trim().startsWith("【提示】"))));
}

function failedResult(m: TurnMessage): boolean {
  return m.isError === true || Boolean(m.details?.error);
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return (content as Block[])
    .filter((b) => b.type === "text" && typeof b.text === "string")
    .map((b) => b.text)
    .join("\n");
}

/**
 * 把这一轮的消息整理成验收用的状态：请求、按顺序的「工具(参数) → 结果」、最终回复。
 * `turnMessages` 是本轮新增的消息（用户消息之后的助手 / 工具结果消息）。
 */
export function buildVerifyState(request: string, turnMessages: AgentMessage[]): { context: string; reply: string } {
  const actions: string[] = [];
  let reply = "";
  for (const m of turnMessages as TurnMessage[]) {
    if (m.role === "assistant" && Array.isArray(m.content)) {
      for (const b of m.content as Block[]) {
        if (b.type === "toolCall") actions.push(`${b.name}(${JSON.stringify(b.arguments ?? {})})`);
      }
      const text = textOf(m.content).trim();
      if (text) reply = text;
    } else if (m.role === "toolResult" && actions.length) {
      const raw = resultTextOf(m.content);
      const result =
        m.toolName === "ask_user" && isUnansweredAsk(raw, failedResult(m))
          ? ASKED_AND_WAITING
          : raw.replace(/\s+/g, " ").trim().slice(0, RESULT_CHARS);
      actions[actions.length - 1] += ` → ${result || "(empty)"}`;
    }
  }
  const list = actions.length ? actions.map((a, i) => `${i + 1}. ${a}`).join("\n") : "(none)";
  return { context: `User request: ${request}\n\nActions taken:\n${list}`, reply: reply.slice(0, REPLY_CHARS) };
}

/**
 * 这一轮是不是「只做了查找，而且全都没找到」（知识库为空、网页不存在……）。
 * 这时验收不通过也不升级：云端模型用的是同一批工具，同样查不到，升级只会多花一次云端解码
 * （E6：网页不存在 0.25、知识库为空 0.44 / 0.75，升级后都没做出来）。本地如实说「没找到」就是对的结局。
 */
export function lookupsAllEmpty(turnMessages: AgentMessage[]): boolean {
  let lookups = 0;
  for (const m of turnMessages as TurnMessage[]) {
    if (m.role === "assistant" && Array.isArray(m.content)) {
      for (const b of m.content as Block[]) {
        if (b.type !== "toolCall") continue;
        if (!b.name || !SEARCH_TOOLS.has(b.name)) return false;
        lookups += 1;
      }
    } else if (m.role === "toolResult") {
      if (!failedResult(m) && !isEmptyResult(resultTextOf(m.content))) return false;
    }
  }
  return lookups > 0;
}

export type VerifyOutcome = { ok: true; p: number } | { ok: false; error: string };

/** 云端 JEV 判一次（noul 的 p = 处理对了的概率）。 */
export async function verifyTurn(request: string, turnMessages: AgentMessage[], signal?: AbortSignal): Promise<VerifyOutcome> {
  const { context, reply } = buildVerifyState(request, turnMessages);
  const result = await runSystemOne(
    {
      state: {
        messages: [
          { role: "user", content: context },
          { role: "assistant", content: reply || "(no reply)" },
        ],
      },
      model: SYSTEMONE_DEFAULT_MODEL,
      questions: { done: { type: "noul", instructions: VERIFY_QUESTION } },
    },
    { signal, forceCloud: true },
  );
  if (!result.ok) return { ok: false, error: result.message };
  const answer = result.response.answers.done;
  if (!answer || answer.type !== "noul") return { ok: false, error: "云端 JEV 没有返回判定" };
  return { ok: true, p: answer.noul };
}

/** 升级时追加给云端模型的要求（接着这一轮往下做，不重做已成功的步骤）。 */
export function escalationPrompt(p: number): string {
  return (
    `【云端验收未通过（判定 ${p.toFixed(2)}）】请检查上面这一轮的处理：补上遗漏的步骤，更正参数或结论里的错误；` +
    "已经成功完成的步骤不要重复。如果缺的是只有用户知道的信息，就直接问用户。最后重新给出完整的答复。"
  );
}

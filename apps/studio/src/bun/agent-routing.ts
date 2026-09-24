/**
 * 「精简路由」工具策略（`AGENT_TOOL_STRATEGY=routed`）的选组逻辑。
 *
 * 每轮开始时，用 JEV（`runSystemOne`，与 JEV 页 / 网关同一条路）判断这一轮需要哪些工具组：
 * 核心工具常驻，只把选中的组追加到工具列表末尾。依据是 E5 实验
 * （docs/experiments/e5-tool-disclosure）：本地 4B 上「核心 + JEV 选组」与全量工具同为 91%，
 * 开局就选全所需组 14/14，而前缀只有全量的 20～40%；让模型自己去找工具只有 79%。
 *
 * JEV 没配 / 超时 / 出错时退回「全部组都加载」：行为等同精简扁平工具集（E5 的 B 组，88%），
 * 不会因为判定服务不可用就少给工具。
 */
import { SYSTEMONE_DEFAULT_MODEL } from "../shared/systemone";
import { ROUTED_GROUPS, ROUTED_GROUP_ORDER, type RoutedGroupId } from "./agent-routed-tools";
import { runSystemOne } from "./systemone";

/** 第二名的概率不低于它时，第二组也一起加载（跨组任务，如「转写录音再发邮件」）。 */
export const ROUTING_SECOND_GROUP_P = 0.25;
/** 选组最多等这么久：本地 JEV 一次前向约 3 秒，超时就退回全部加载，不让用户干等。 */
export const ROUTING_TIMEOUT_MS = 8000;

export type GroupPick = {
  groups: RoutedGroupId[];
  via: "jev" | "fallback";
  probabilities?: Record<string, number>;
  error?: string;
};

/**
 * 概率 → 要加载的组：第一名（不是 none 时）加载；第二名不是 none 且概率 ≥ 阈值也加载。
 * 只认 available 里的组，结果按固定顺序排列（同一组合的工具列表逐字节一致，前缀缓存才能命中）。
 */
export function groupsFromProbabilities(probabilities: Record<string, number>, available: RoutedGroupId[]): RoutedGroupId[] {
  const ranked = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const picked = new Set<string>();
  const [first, second] = ranked;
  if (first && first[0] !== "none") picked.add(first[0]);
  if (second && second[0] !== "none" && second[1] >= ROUTING_SECOND_GROUP_P) picked.add(second[0]);
  return ROUTED_GROUP_ORDER.filter((g) => available.includes(g) && picked.has(g));
}

/** 选组问题的选项：none + 每个可用组一句话。选项文本固定，便于 JEV 侧的前缀缓存。 */
export function groupQuestionCriteria(available: RoutedGroupId[]): Record<string, string> {
  const criteria: Record<string, string> = {
    none:
      "None of these: the request only needs web search, reading or writing files, the user's knowledge base / notes / memory, " +
      "or no tool at all.",
  };
  for (const g of ROUTED_GROUP_ORDER) {
    if (available.includes(g)) criteria[g] = `The request needs the ${ROUTED_GROUPS[g].summary} tools.`;
  }
  return criteria;
}

export async function pickToolGroups(
  request: string,
  available: RoutedGroupId[],
  opts: { attachments?: string[]; signal?: AbortSignal } = {},
): Promise<GroupPick> {
  if (!available.length) return { groups: [], via: "jev" };
  const timeout = AbortSignal.timeout(ROUTING_TIMEOUT_MS);
  const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;
  const files = opts.attachments?.length ? opts.attachments.join(", ") : "none";
  try {
    const result = await runSystemOne(
      {
        state: `Attached files: ${files}\nUser request: ${request}`,
        model: SYSTEMONE_DEFAULT_MODEL,
        questions: {
          group: {
            type: "choice",
            instructions: "Which group of tools does the assistant need to handle this request? Pick the group needed first.",
            criteria: groupQuestionCriteria(available),
          },
        },
      },
      { signal },
    );
    if (!result.ok) return { groups: [...available], via: "fallback", error: result.message };
    const answer = result.response.answers.group;
    if (!answer || answer.type !== "choice") return { groups: [...available], via: "fallback", error: "no choice answer" };
    return { groups: groupsFromProbabilities(answer.probabilities, available), via: "jev", probabilities: answer.probabilities };
  } catch (e) {
    return { groups: [...available], via: "fallback", error: e instanceof Error ? e.message : String(e) };
  }
}

/**
 * 附在本轮用户消息末尾的一段说明：还有哪些组没加载、怎么加载。
 * 放用户消息而不是系统提示：系统提示是请求最靠前的部分，每轮改它会让前缀缓存整段作废。
 */
export function routedTurnNote(notLoaded: RoutedGroupId[]): string | null {
  if (!notLoaded.length) return null;
  const lines = notLoaded.map((g) => `- ${ROUTED_GROUPS[g].summary}`).join("\n");
  return (
    "--- 其他工具组（尚未加载） ---\n" +
    `${lines}\n` +
    "当前工具做不了用户要的事时，先调用 load_tools 加载对应的组。\n--- 说明结束 ---"
  );
}

/** 「记住…」类请求的说法（中英文）。 */
export const REMEMBER_INTENT = /记住|记下|记一下|帮我记|别忘了|不要忘了|以后.*(注意|记得)|remember|don't forget|keep in mind|note that/i;

/** 小模型缺信息时常填的占位值：尖括号 / 花括号模板、TODO、unknown、问号、xxx 等。 */
const PLACEHOLDER = /^\s*(<[^<>]{0,40}>|\{\{[^{}]{0,40}\}\}|\[[^[\]]{0,40}\]|todo|tbd|unknown|placeholder|n\/a|x{3,}|\?+|？+|待定|未知|某人|收件人)\s*$/i;
/** 这些工具的字符串参数本来就可能是占位写法（补丁 / 命令 / 文件正文），不查。 */
const FREEFORM_TOOLS = new Set(["apply_patch", "bash", "write_file", "edit_file", "ask_user"]);

/**
 * 精简路由策略的参数兜底：参数里出现明显的占位值时拦下这次调用，返回给模型的原因里让它先 ask_user。
 * 依据 E1 / E5：小模型信息不足时很少主动反问，而是填一个占位值硬做（「给他发封邮件」→ to="<收件人>"）。
 * 只看顶层字符串参数。空字符串不算：模型常给可选参数传 ""（如 negative_prompt），那是正常写法。
 */
export function routedArgProblem(toolName: string, args: Record<string, unknown>): string | null {
  if (FREEFORM_TOOLS.has(toolName)) return null;
  for (const [key, value] of Object.entries(args)) {
    if (typeof value !== "string") continue;
    if (PLACEHOLDER.test(value)) {
      return (
        `参数 ${key} 的值「${value}」是占位符，不是真实信息。` +
        "如果这是只有用户知道的信息，先用 ask_user 问用户；如果能从之前的工具结果里得到，就用那个真实值重新调用。"
      );
    }
  }
  return null;
}

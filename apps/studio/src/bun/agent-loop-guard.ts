/**
 * 精简路由策略的「本轮循环守卫」：拦住本地小模型的两种原地打转。
 *
 * E6（docs/experiments/e6-real-app，真实 App + 本地 Qwen3.5-4B）里观察到：
 * 1. **搜索打转**：知识库为空 / 网页不存在时，模型换着关键词把 recall、find、note_read、web_search、
 *    web_fetch 轮番调十几次（一题 80～200 秒），始终不肯承认没找到；
 * 2. **生成失败后反复重试**：生图 / 语音后端没配好，工具已经明说「本轮不要重试」，
 *    模型仍然再调一次，或改用 bash、load_tools 绕路，或连续追问用户。
 *
 * 规则（每轮用户消息开始时重置）：
 * - 搜索类工具：连续 2 次没结果、或本轮已搜 5 次 → 在工具结果末尾追加「停止搜索、直接回答」的提示；
 *   连续 3 次没结果、或本轮已搜 8 次 → 直接拦下后续搜索。
 * - 生成类工具失败后：同一个生成工具本轮不再允许调用；bash / load_tools 这类绕路手段也拦下。
 *
 * 纯逻辑，无副作用；接线在 agent.ts 的 beforeToolCall / afterToolCall。
 */

/** 查找类工具（只读、会被模型换关键词反复调用的那些）。 */
export const SEARCH_TOOLS = new Set([
  "recall",
  "knowledge_search",
  "note_search",
  "note_list",
  "note_read",
  "memory_search",
  "web_search",
  "web_fetch",
  "find",
  "glob",
  "grep",
  "list_dir",
  "media_search",
]);

/** 生成类工具（失败通常是后端 / 配置问题，重试没有意义）。 */
export const GENERATION_TOOLS = new Set(["generate_image", "generate_speech", "generate_video", "generate_music"]);

/** 生成失败后本轮拦下的绕路工具。 */
const WORKAROUND_TOOLS = new Set(["bash", "load_tools", "apply_patch"]);

export const SEARCH_NOTICE_EMPTY_STREAK = 2;
export const SEARCH_NOTICE_TOTAL = 5;
export const SEARCH_BLOCK_EMPTY_STREAK = 3;
export const SEARCH_BLOCK_TOTAL = 8;

/** 「没找到」的常见说法（各检索工具的空结果文案 + 模型常见的英文空结果）。 */
const EMPTY_RESULT =
  /nothing found|no match|no matching|no relevant|no results?\b|not found|0 results|没搜到|没有找到|未找到|没有匹配|无结果|找不到|不存在|page not found|404/i;

export function isEmptyResult(text: string): boolean {
  const t = text.trim();
  return t.length === 0 || (t.length < 600 && EMPTY_RESULT.test(t));
}

export class TurnLoopGuard {
  private searches = 0;
  private emptyStreak = 0;
  private readonly failedGeneration = new Map<string, string>();

  /** 调用前检查：返回拦截原因（给模型看的），null = 放行。 */
  check(toolName: string): string | null {
    if (SEARCH_TOOLS.has(toolName) && (this.emptyStreak >= SEARCH_BLOCK_EMPTY_STREAK || this.searches >= SEARCH_BLOCK_TOTAL)) {
      return (
        `本轮已经搜索 ${this.searches} 次${this.emptyStreak ? `，最近连续 ${this.emptyStreak} 次没有结果` : ""}，不再继续搜索。` +
        "请根据已有结果直接回答；确实没有相关信息就如实告诉用户没找到，需要用户补充时用 ask_user 问一次。"
      );
    }
    const failed = this.failedGeneration.get(toolName);
    if (failed !== undefined) {
      return `本轮 ${toolName} 已经失败过（${failed}），重试不会成功。请直接告诉用户失败原因和解决办法，不要再调用。`;
    }
    if (this.failedGeneration.size && WORKAROUND_TOOLS.has(toolName)) {
      const [tool, reason] = [...this.failedGeneration.entries()][0]!;
      return `${tool} 失败（${reason}）后不要改用 ${toolName} 绕路。请直接告诉用户失败原因和解决办法，然后结束这一轮。`;
    }
    return null;
  }

  /** 调用后记账：返回要追加在工具结果末尾的提示（给模型看的），null = 不追加。 */
  record(toolName: string, resultText: string, failed: boolean): string | null {
    if (GENERATION_TOOLS.has(toolName) && failed) {
      this.failedGeneration.set(toolName, resultText.trim().slice(0, 120) || "未知原因");
      return "【提示】生成失败通常是后端或配置问题，本轮不要重试、也不要换别的工具绕路：直接告诉用户失败原因和解决办法。";
    }
    if (!SEARCH_TOOLS.has(toolName)) return null;
    this.searches += 1;
    this.emptyStreak = failed || isEmptyResult(resultText) ? this.emptyStreak + 1 : 0;
    if (this.emptyStreak >= SEARCH_NOTICE_EMPTY_STREAK || this.searches >= SEARCH_NOTICE_TOTAL) {
      return (
        `【提示】本轮已搜索 ${this.searches} 次${this.emptyStreak ? `，连续 ${this.emptyStreak} 次没有结果` : ""}。` +
        "不要再换关键词重复搜索：根据已有结果回答；确实没有就如实告诉用户没找到。"
      );
    }
    return null;
  }
}

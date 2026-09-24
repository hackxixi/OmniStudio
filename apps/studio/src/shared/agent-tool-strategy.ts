/**
 * Agent 工具调用方式（设置 → Agent 能力）：
 * - classic（经典，默认）：全部工具一次给模型，行为与历史版本一致；
 * - routed（精简路由）：常驻核心工具 + 按需加载工具组（创作 / 开发 / 已连接应用），
 *   适合本地小模型 —— 前缀更短、首轮更快。
 *
 * 策略的具体接线在 Agent 侧（agent.ts / agent-context.ts），本文件只定义设置值
 * 的取值与归一，供 RPC 与设置页共用。
 */
export type AgentToolStrategy = "classic" | "routed";
export const AGENT_TOOL_STRATEGIES: readonly AgentToolStrategy[] = ["classic", "routed"];

/** 设置值 → 策略；空、未知值一律 classic（老用户升级后行为不变）。 */
export function parseToolStrategy(value: unknown): AgentToolStrategy {
  return value === "routed" ? "routed" : "classic";
}

/**
 * AgentTypeBadge 组件测试：渲染出三种品牌的胶囊与 logo，null 时不渲染，
 * withLabel 时带文字。
 */
import { describe, expect, test } from "bun:test";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import { AgentTypeBadge } from "./agent-type-badge";

describe("AgentTypeBadge", () => {
  test("claude：品牌色 + 浅底胶囊 + logo svg", () => {
    const html = renderToString(createElement(AgentTypeBadge, { type: "claude", size: 11 }));
    expect(html).toContain("d97757");
    expect(html).toContain("rgba(217, 119, 87, 0.18)");
    expect(html).toContain("viewBox=\"0 0 24 24\"");
    expect(html).toContain("title=\"Claude\"");
  });

  test("codex：OpenAI 螺旋 + 绿色胶囊", () => {
    const html = renderToString(createElement(AgentTypeBadge, { type: "codex", size: 11 }));
    expect(html).toContain("10a37f");
    expect(html).toContain("rgba(16, 163, 127, 0.18)");
  });

  test("pi：块状 Pi 字母（三个 path）+ teal 胶囊", () => {
    const html = renderToString(createElement(AgentTypeBadge, { type: "pi", size: 11 }));
    expect(html).toContain("8abeb7");
    expect(html).toContain("rgba(138, 190, 183, 0.18)");
    // 官方 logo 的三块
    expect(html).toContain("M165.29 165.29H517.36V400H400V282.65H165.29Z");
    expect(html).toContain("M517.36 400H634.72V634.72H517.36Z");
  });

  test("withLabel：带全名文字 svg", () => {
    const html = renderToString(createElement(AgentTypeBadge, { type: "claude", withLabel: true }));
    expect(html).toContain(">Claude</text>");
  });

  test("null / 未知值：不渲染", () => {
    expect(renderToString(createElement(AgentTypeBadge, { type: null }))).toBe("");
    expect(renderToString(createElement(AgentTypeBadge, { type: undefined }))).toBe("");
    expect(renderToString(createElement(AgentTypeBadge, { type: "gemini" as any }))).toBe("");
  });
});

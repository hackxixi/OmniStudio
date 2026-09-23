import { describe, expect, test } from "bun:test";

import { orderGitRemotes } from "./installer";

/**
 * 技能仓库 clone 的多链路顺序：按候选顺序短超时探路，第一条答得上的排最前，
 * 其余保持原序留给 clone 失败时换。
 */
describe("orderGitRemotes", () => {
  const direct = "https://github.com/anthropics/skills.git";
  const proxy = `https://gh-proxy.com/${direct}`;
  const fast = `https://ghfast.top/${direct}`;

  test("首条就通：原序不变，只探一次", async () => {
    const probed: string[] = [];
    const order = await orderGitRemotes([direct, proxy, fast], async (r) => {
      probed.push(r);
      return true;
    });
    expect(order).toEqual([direct, proxy, fast]);
    expect(probed).toEqual([direct]);
  });

  test("直连探不通（国内的 75s 连接超时被短超时截断）：第一条通的镜像提到最前", async () => {
    const order = await orderGitRemotes([direct, proxy, fast], async (r) => r === fast);
    expect(order).toEqual([fast, direct, proxy]);
  });

  test("全都探不通：保持原序，交给 clone 报真实错误（仓库不存在 / 私有）", async () => {
    expect(await orderGitRemotes([proxy, direct], async () => false)).toEqual([proxy, direct]);
  });

  test("只有一个候选（非 GitHub 远端）：不探路", async () => {
    let probed = 0;
    const order = await orderGitRemotes(["git@gitlab.com:a/b.git"], async () => {
      probed += 1;
      return false;
    });
    expect(order).toEqual(["git@gitlab.com:a/b.git"]);
    expect(probed).toBe(0);
  });
});

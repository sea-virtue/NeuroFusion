# SlipTack 2024：来源核查

核查日期：2026-11-11。用户指定 DOI：10.1109/LRA.2024.3400148。

论文：Y. Gloumakov, T. M. Huh, H. S. Stuart, *Fast In-Hand Slip Control on Unfeatured Objects with Programmable Tactile Sensing*, IEEE RA-L 9(7), 6059–6066, 2024。

## 当前状态

**没有找到可核实的官方源码仓库；未克隆代码，未安装依赖，未连接硬件。** 这不是已经导入的上游算法实现。不可用另一篇 slip-aware manipulation 的仓库替代并标注为本篇作者代码。

- 作者论文页：https://edg.berkeley.edu/publication/fast-in-hand-slip-control-on-unfeatured-objects-with-programmable-tactile-sensing/
- 作者实验室演示：https://edg.berkeley.edu/research/tactile-sensing/
- 作者视频：https://www.youtube.com/watch?v=MKm3UeJY-y8
- 开放论文：https://escholarship.org/content/qt9k30z443/qt9k30z443.pdf
- GitHub 公共仓库名称/描述检索 `SlipTack`、`"Fast In-Hand Slip"`、`Gloumakov` 均返回 0 条。这不证明不存在未索引、不同名称或未公开的仓库。
- 作者论文页及已读取论文正文没有提供已核实的源码入口。论文明确提到 Python 实验脚本，但脚本存在不等于脚本公开。
- PDF 已通过网页阅读工具核对；WSL 和 Windows 的直接下载均返回 HTTP 403，本次未保存本地 PDF。

## 方法适配边界

1. 原传感器是带机械凸起的电容触觉 SlipTack；其信号—加速度回归和粘滑频率关系依赖传感器、材料及标定，不能把事件计数直接代入。
2. PDF 第 6 页（印刷页 5）IV-E：利用约 0.03 s 的触觉信号估计加速度、确定开启时长，并补偿约 0.12 s 的名义电机响应；目标下滑距离为 20/25/30 cm。这是该演示的预测式再抓取，不是已实现的连续目标位置反馈控制。
3. 可借鉴“运动估计—响应补偿—目标距离再抓取”的教师结构；事件触觉运动估计、夹爪制动模型、动作映射需在本平台重新标定与验证。
4. 后续若独立实现，应放在 `src/neurofusion/teachers/` 等本项目目录并写明论文启发，不能称作作者原码。`third_party` 只放来源可验证的上游代码。

采集方案见 `docs/slip-bc-collection.md`。本条记录使用独立 reference JSON，不加入 `sources.lock.json`，避免现有 `fetch_upstream.py` 将未找到仓库的参考资料误作可恢复的 Git checkout。

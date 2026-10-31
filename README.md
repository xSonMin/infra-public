# infra-public

推理基础设施与 LLM 推理性能研究的公开进展记录。当前研究围绕 DeepSeek-V4-Flash、SGLang 和四张 RTX PRO 6000，比较 MTP、调度间隔、等待预算及 prefill 分块在不同负载下的效果。

公开快照日期：**2026-10-31**。模型、软件版本、硬件范围、请求设置和测量口径以各阶段报告为准。结果均有适用范围，实测现象、机制推断与尚未验证的方向分别记录。

## 研究进展

- [预填充分块 2048/1024 对照（最新阶段）](REPORT-CHUNK-2026-10-31.md)
- [长输入时间线诊断](REPORT-TIMELINE-2026-10-31.md)
- [等待预算原型](REPORT-WAIT-BUDGET-2026-10-31.md)
- [MTP 步数与调度间隔初筛](REPORT-FACTORIAL-2026-10-31.md)
- [夜间重复验证与负载边界](REPORT-OVERNIGHT-2026-10-31.md)
- [首 token 延迟归因](REPORT-TTFT-ATTRIBUTION.md)
- [持续负载下的调度实验](REPORT-SUSTAINED-SCHEDULING.md)
- [有效吞吐与请求达标率](REPORT-GOODPUT.md)
- [MTP 性能边界](REPORT-MTP-BOUNDARY.md)
- [部署与基础基准](REPORT.md)

## 阅读与使用

- `scripts/`：实验、统计复算与诊断代码。
- `tests/`：不需要连接研究服务器的本地单元测试。
- `docs/superpowers/specs/` 与 `docs/superpowers/plans/`：研究设计及执行方案。
- `results/`：选定的脱敏汇总、阶段完成记录与四张阶段性图表；不含完整原始数据。
- [公开材料说明](PUBLICATION.md)：脱敏范围、证据边界及复现要求。

```bash
python3 -m unittest discover -s tests -v
python3 tools/check_publication.py
```

服务器地址、个人路径和硬件唯一标识已替换为占位值。`scripts/launch.py` 的设备白名单保留了校验逻辑，但 UUID 为匿名占位值；在自己的授权环境运行前，必须核对并配置相应设备及依赖。文档中的 `$HOME`、`$LOCAL_HOME`、`research-user` 和 `gpu-node` 同样表示需自行适配的环境信息。

本仓库发布的是研究材料快照。阶段记录中的服务恢复或历史测试结果不代表仓库发布时重新检查过服务器；完整独立复算还需要原始请求记录和 trace。

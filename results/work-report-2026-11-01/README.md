# 长期报告：2026-11-01 证据与图表

本目录支持根目录 [长期报告](../../REPORT-PROGRESS.md) 的详细修订。最新完整实验的证据截止为北京时间 **2026-11-01 06:02:47**，采用第二次独立执行的定稿，不包含新启动的 GPU 测量，也不代表写作时刻的远端实时验收。

`report-data.json`保存绘图与表格所用派生数值、独立重复信息和来源SHA256。部分路径标识私下保留的原始记录；公开派生汇总不足以独立复现原始实验。`sources/`保留本次采用的阶段报告版本及改写前长期报告，用于工作目录内追溯；公开更新只选择最终报告、派生JSON和已审查图片，不复制整个目录。

## 图表索引

| 图 | 内容 | 统计与范围 |
|---|---|---|
| [1 吞吐曲线](figures/01-baseline-throughput.png) | 三种长度下普通解码/MTP3的C1/8/32输出吞吐 | 主基准54窗口；三轮均值，阴影为三轮最小值至最大值。 |
| [2 延迟曲线](figures/02-baseline-latency.png) | 相同主基准的TTFT与TPOT | 与图1重叠，非新增实验；均值不替代分位数。 |
| [3 生成速度与系统吞吐](figures/03-generation-throughput-tradeoff.png) | 横轴1000/平均TPOT，纵轴四卡整体吞吐/4 | 横轴不是每请求速率算术均值；不能以C乘横轴还原纵轴。 |
| [4 阶段GPU活动](figures/04-gpu-stage-activity.png) | rank0普通decode/target verify、draft、draft_extend | 独立profile窗口；活动并集/CPU注释次数，不是可相加的请求延迟。 |
| [5 到达节奏控制](figures/05-ttft-arrival-control.png) | 单请求、同时32与周期4req/s的TTFT | 27窗口、873请求；单请求每配置仅3个样本。 |
| [6 有效吞吐](figures/06-goodput-arrival-rate.png) | 4/7/9req/s的goodput与达标率 | 27窗口；含排空，拒绝留在达标分母；阴影为观测范围。 |
| [7 门槛敏感性](figures/07-slo-sensitivity.png) | 600秒短输入复验的TTFT门槛变化 | 同一批请求测后复算；TPOT目标固定30ms，不替换主门槛。 |
| [8 ABBA分块成本](figures/08-chunk-abba.png) | 第二次独立执行的四个服务block | 每配置2个block，每点16请求；客户端计时，不是纯GPU成本。 |
| [9 完整负载矩阵](figures/09-night-load-matrix.png) | 三配置×四负载的TTFT、TPOT与达标率 | 最新36个600秒负载窗，三轮服务窗口等权；面板颜色各自归一。 |
| [10 到达分钟cohort](figures/10-night-minute-cohorts.png) | 三轮长输入周期4req/s逐分钟TPOT | 按计划到达分钟选请求、追踪完整生命周期，不是瞬时GPU速度。 |

三轮观测范围不是置信区间。图1、2、3及单流复核是同一批主基准的不同展示，图9、10也重叠，不能重复累计请求数。先前日报三张图继续保留，本次新增的十张图单独编号。

## 重新绘图

在有 Matplotlib、NumPy、Pillow 和可用中文字体的环境中，从仓库根目录运行：

```bash
python3 tools/render_progress_figures.py
```

本次使用 Matplotlib3.11.2、Pillow13.0.0，图片为PNG；脚本只读本目录JSON，写入`figures/`，不访问网络或模型服务。PNG文本/EXIF元数据均移除。不同字体可能改变布局，发布前仍需视觉审查。

## 核验记录

- 54份主基准原始JSONL的实际输出token、成功数、吞吐公式及TPOT公式已在本次整理核对。
- 最新40个正式case的原始请求文件SHA256与独立审计一致；最终阶段报告hash与完成记录一致。
- 第二次执行最终40case、75,236计划、73,291成功、1,945本地拒绝，未预期错误0；预热328和smoke8单列。
- 第一次失败不混入第二次；第三轮one2048的性能变化与未解释机制保留。
- 源摘要验证证明本次使用的文件一致，不表示重新运行GPU实验；历史恢复记录不等于当前实时健康检查。

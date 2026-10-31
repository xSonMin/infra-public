> 公开脱敏副本；环境标识已替换，原始证据另行保留。详见仓库根目录 PUBLICATION.md。

# Long-input Timeline Diagnostic Implementation Plan

> **For agentic workers:** Execute the authorized scope sequentially. No delegation was requested.

**Goal:** 在固定MTP1步的三个调度条件下，对长输入采集有限时间线并完成保守归因。

**Architecture:** 沿用已验证的32在途原生客户端及TP4等待预算包装。新目录加载器显式限定本轮范围，独立驱动按客户端时钟40秒触发48步profile。每配置先无采样对照，再采样；数据与raw trace按文件hash回收并独立分析。

**Tech Stack:** Python标准库、原安装版SGLang/PyTorch profiler、原四GPU。

## Global Constraints

- 固定1步，间隔0/8、100ms预算预先冻结；三组均含相同CPU广播。
- 6个120秒窗口，2048/128@4req/s，共2880个正式计划请求；最多32在途。
- 采样48步、12份rank trace、压缩trace上限1GiB；关闭stack与shape采集。
- 请求超时45秒、case300秒；100ms发送偏差不放宽；全程记录profiling扰动。
- 新截止90分钟并预留恢复15分钟，恢复原3步/间隔0且无覆盖环境。

### Task 1: Capture and trace accounting

- [x] 对profile payload、目标目录、阶段标签解析、DRAFT排除、异步launch关联与空GPU trace写失败测试。
- [x] 实现只读trace汇总和固定48步采样设置，验证阶段与批大小口径。

### Task 2: Isolated driver and client

- [x] 测试三配置同MTP深度及广播覆盖、干净恢复、固定长输入及发送时钟检查。
- [x] 新客户端独立保存计时原点，驱动在40秒调用profiler，API不阻塞客户端事件循环。
- [x] 固定代码/输入/到达摘要、6窗口顺序、90分钟截止及trace总量上限。

### Task 3: Execute, audit and restore

- [x] 核验原服务/进程/锁，原服务8请求冒烟，各配置质量和四rank回执。
- [x] 完成三个无采样对照与三个诊断窗口，保存profiler API及12份rank原始trace。
- [x] 独立重算客户端计量和trace摘要，区分直接观察、机制推断和未识别活动。
- [x] 恢复原服务/源码/环境、回收原始文件并校验，写真实完成状态及报告。

最终状态：completed。6窗口2880正式计划请求、12份rank原始trace完成，本地/远端115项测试和独立审计规则4项测试通过，原服务恢复独立验收通过。采样结束重复stop的三次500及传输超时记录保留。

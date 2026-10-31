> 公开脱敏副本；环境标识已替换，原始证据另行保留。详见仓库根目录 PUBLICATION.md。

# Prefill Chunk Comparison Implementation Plan

> **For agentic workers:** Execute this authorized stage sequentially. No delegation was requested.

**Goal:** 在固定等待预算下测量2048/1024分块的取舍，核对输出、真实分块和原服务恢复。

**Architecture:** 新驱动复用已测客户端与GPU trace解析，命令只更改原生chunked-prefill-size值。每配置各做无采样/采样窗口，采样自动停止按完整rank trace验收。独立请求、输出与trace审计，本地复算并保留原始证据。

**Tech Stack:** 原SGLang/PyTorch profiler、Python标准库、原四GPU与原冻结输入。

## Global Constraints

- MTP1步、interval8、100ms等待预算及协调广播固定，唯一配置因素为chunk2048/1024。
- 4窗口1920正式请求，120秒、2048/128@4req/s、最大32在途、100ms发送偏差。
- 请求45秒、client case300秒；48-step诊断×两配置×四rank，压缩总量上限512MiB。
- 安装源码不变；新运行目录与90分钟截止，预留恢复15分钟，恢复原3步/interval0/chunk2048。

### Task 1: Correct configuration and bounded profiler lifecycle

- [x] 编写命令唯一差异、原样恢复/环境清理、loader范围、chunk参数合法性和自动停止完整文件判定的失败测试。
- [x] 实现新分块驱动/loader/worker和48步profile自动停止分类，保留异常响应。
- [x] 验证本地全部测试，冻结长输入正确性探针与文本一致性门槛。

### Task 2: Remote preflight and execution

- [x] 核实锁/PID/GPU/服务/原始源码，远端测试与原服务8请求冒烟。
- [x] 每配置核对四rank覆盖回执及4条长输入生成，核验真实分块参数、输出token与retraction。
- [x] 完成4个有限窗口、8份rank trace，逐窗checkpoint。

### Task 3: Audit and restore

- [x] 独立审计逐请求、分块参数/文本一致性及每rank真实EXTEND token量。
- [x] 原服务恢复命令/环境/源码/GPU/健康/真实生成独立验收。
- [x] 原始文件分批回收并校验，8份trace本地复算，报告全窗口与中心队列及真实完成范围。

最终状态completed：4窗口1920正式计划请求及8份rank原始trace全部完成，两配置各4条长输入文本一致，本地和远端各122项测试通过，原3步/interval0/chunk2048服务恢复独立验收通过。1024候选性能退化，保留负结果且不推广。

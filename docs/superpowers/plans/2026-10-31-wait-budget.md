> 公开脱敏副本；环境标识已替换，原始证据另行保留。详见仓库根目录 PUBLICATION.md。

# Waiting Budget Prototype Implementation Plan

> **For agentic workers:** Use executing-plans to implement the approved scope sequentially. No subagent delegation was requested.

**Goal:** 在不改变安装目录源码的前提下，对四卡TP4调度增加100毫秒等待预算，并完成可恢复的有限初筛。

**Architecture:** 本轮覆盖加载器只包装固定版本Scheduler方法。协调rank执行原始轮数逻辑并依据等待预算决定是否放行，三个整数经CPU TP组广播后更新全部rank。客户端与服务切换沿用已验证入口，新实验使用独立目录与新冻结协议。

**Tech Stack:** Python标准库、已安装SGLang和PyTorch Gloo，原四卡与localhost服务。

## Global Constraints

- 仅原项目、GPU0～3、DeepSeek-V4-Flash与127.0.0.1:30000。
- 预算固定100ms，仅主动延后，研究SLO保持TTFT500ms/TPOT30ms。
- 单轮8窗口5280正式计划请求，截止前预留恢复时间，异常恢复原命令。
- 原安装源码不修改，恢复环境不能带入实验覆盖。

### Task 1: Pure policy and transparent loader

- [x] 为预算到期、无等待请求、无效时钟/时间戳、禁用预算、恢复路径和摘要不匹配编写失败测试。
- [x] 实现scripts/wait_budget_policy.py和本轮显式加载器，测试禁用状态原行为不变。
- [x] 验证加载器只包装固定目标模块，安装源码字节保持一致。

### Task 2: Four-process consensus

- [x] 为协调rank权威决策、从属rank时钟偏差与一致剩余轮数写测试。
- [x] 在远端原项目以四个CPU进程、文件式rendezvous运行有超时的Gloo测试，不使用GPU。
- [x] 多卡一致性已由四进程CPU Gloo确认，进入冻结GPU初筛。

### Task 3: Bounded experiment and smoke

- [x] 测试独立运行目录、四配置命令、覆盖环境显式启用与默认恢复清理。
- [x] 冻结代码/输入/到达摘要、执行顺序、8个正式窗口和新的90分钟截止。
- [x] 原服务8请求冒烟通过，两组覆盖配置的四rank回执及各配置正确性均通过，8窗口已执行。

### Task 4: Independent audit and restoration

- [x] 逐请求独立审计完成，分列仅同步差异和预算差异，本地复算一致。
- [x] 原命令和干净环境已恢复并独立验收，主进程及四GPU进程均核验。
- [x] 全部原始窗口按摘要回收并本地复算，102项本地/远端测试通过，状态completed，报告完成。

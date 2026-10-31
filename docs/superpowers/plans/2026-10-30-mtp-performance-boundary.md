> 公开脱敏副本；环境标识已替换，原始证据另行保留。详见仓库根目录 PUBLICATION.md。

# MTP 性能边界与瓶颈研究 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 在当前会话顺序执行；未获得新的明确授权前不使用子代理。

**Goal:** 产出可追溯的《MTP 性能边界与瓶颈报告》，解释收益来源、适用边界，并验证一个有证据支持的改进假设。

**Architecture:** 复用现有 UUID 安全启动器、停止脚本和 SGLang 原生压测，不另建通用调度平台。先取得独立、无 profiler 的普通解码/MTP 基准，再以少量代表点剖析各 TP rank 的执行时间线；根据证据筛选固定推测深度和现有 adaptive 功能，独立复验后编写报告。

**Tech Stack:** Python 3.12（远端）、Python unittest、SGLang 0.5.23、PyTorch 2.14.1、FlashInfer 0.7.2、Nsight Systems、标准库 JSON/CSV/hashlib。

## Global Constraints

- 依据：`$LOCAL_HOME/Documents/infra/docs/superpowers/specs/2026-10-30-mtp-performance-boundary-design.md`，书面 spec 已获用户确认。
- 本地根目录 `$LOCAL_HOME/Documents/infra`；远端根目录 `$HOME/infra-deepseek-v4`；SSH 使用已有配置项 `gpu-node`，单连接检查，不复制私钥。
- 模型固定为 `deepseek-ai/DeepSeek-V4-Flash`，revision `60d8d70770c6776ff598c94bb586a859a38244f1`；固定安装版本，不为使用在线文档的新特性升级依赖。
- 只使用物理 GPU 0、1、2、3，并继续通过 UUID 校验；第一阶段保持 TP=4、后端 `flashinfer_mxfp4`、关闭 radix cache、绑定 `127.0.0.1:30000`。
- 保持 context length=8192、chunked prefill size=2048、max running requests=32、CUDA Graph decode batch 上限=32；诊断性改变必须另立 case。
- 不修改驱动、权重、接受阈值或其他用户进程；只停止项目记录且 UID、命令和 UUID 均核验通过的服务。
- 不覆盖此前的 `REPORT.md`、`results/final-summary.json`、远端原始快照或已有实验 JSONL。
- 新结果置于 `results/mtp-study-2026-10-30/`；每次执行创建独立 `run-YYYYMMDDTHHMMSSZ` 子目录，已存在即拒绝覆盖。
- 正式计时至少三轮；约 5% 的变化至少五轮独立复核；单轮独立 pilot 选择配对请求数，使较快配置计时也达到约 30 秒。
- Profiler 数据只用于归因，不替代无 profiler 的吞吐验收；通信是阶段内部活动，不得与阶段 wall time 重复相加。
- 当前不是 Git 仓库：不初始化 Git、不创建 worktree、不假称 commit；以逐任务校验记录和 SHA-256 保存检查点。

## 本轮状态与执行入口

本计划是执行前文档；复选框只在相应原始文件、测试或日志核验后勾选。不因计划写成而将实验标成完成。后续执行直接从 Task 1 开始，不再次要求用户确认已经批准的研究目标和 spec。

## 文件职责

| 文件（均相对本地/远端项目根目录） | 动作 | 单一职责 |
|---|---|---|
| `scripts/bench_suite.py` | 小幅扩展 | 可指定正式请求数、warmup 数、seed、结果目录；保留旧默认行为 |
| `scripts/benchmark_worker.py` | 新建 | 通过已安装 SGLang 客户端记录精确 token 请求清单和请求级指标 |
| `tests/test_benchmark_result.py` | 增补 | 验证逐请求长度、错误列表、数值有效性与请求规模 |
| `tests/test_benchmark_worker.py` | 新建 | 验证 token 清单顺序、哈希和文本输入拒绝 |
| `scripts/launch.py` | 小幅扩展 | 固定 1/2/3 步及受审查 adaptive 配置入口，保持 GPU 安全校验 |
| `scripts/mtp_experiment.py` | 新建 | 串行 pilot/正式实验、安全停启、GPU遥测、配对和恢复 |
| `tests/test_launch.py` | 增补 | 新参数和默认命令的回归检查 |
| `tests/test_mtp_experiment.py` | 新建 | 验证 case 规模、校准、目录边界、进程祖先和配对 |
| `scripts/mtp_analysis.py` | 新建 | 只做离线配对统计与时间区间并集计算，不连接或修改服务 |
| `scripts/mtp_profile.py` | 新建 | 从实际请求进度触发服务端短窗口采集，校验四个 rank 与 launch correlation |
| `scripts/mtp_tuning.py` | 新建 | 必须取得证据决策后才运行深度筛选、冻结策略及独立复验 |
| `tests/test_mtp_analysis.py` | 新建 | 配对、缺失数据、累计指标及重叠时间归因 |
| `tests/test_mtp_profile.py` | 新建 | GPU trace有效性、异步归因与采集时机 |
| `tests/test_mtp_tuning.py` | 新建 | 受限候选和独立重复要求 |
| `sources/benchmark-prompts-validation.json` | 新建 | 冻结的未参与筛选的验证提示集，hash随策略记录 |
| `results/mtp-study-2026-10-30/run-*/` | 新建 | 快照、原始数据、profile、冻结配置、汇总与 SHA-256 清单 |
| `REPORT-MTP-BOUNDARY.md` | 新建 | 证据分级、性能边界、假设检验与负结果 |
| `README.md` | 追加 | 研究报告及复现入口，不改既有验收结论 |

---

### Task 1: 预检、固定证据边界和恢复方案

**Files:** Read `scripts/launch.py`, `scripts/stop.py`, `results/mtp-study-2026-10-30/source-audit.json`; create run directory 的 `preflight/`、`commands.log` 和 `run-manifest.json`。

**Interfaces:** 输入现有 SSH 配置与 active-server 记录；输出 UTC run_id、原服务 PID/命令/启动参数、GPU UUID 白名单、固定依赖和 profiler 能力；不生成请求负载。

- [x] 单连接读取 `id -un; hostname; pwd`，然后读取本项目 active-server 记录、进程 UID/cmdline、四卡 UUID/显存和 `/health`、`/v1/models`；不认为旧 PID 永久有效。
- [x] 保存安装版 CLI 和源码证据：benchmark 的请求数/seed/profile 开关、`ProfileReq` 请求结构、`/start_profile` 和 `/stop_profile` handler、adaptive 配置解析器、accept 计数读取与重置点。
- [x] 创建独立 run 目录，以 `x` 模式写 manifest；记录模型分片 manifest、提示集、脚本和源码 SHA-256，保存 Python/package 版本、GPU 驱动、温度/频率/功耗和磁盘空间。
- [x] 保存原服务的重启命令和安全停止步骤；若项目服务外存在占用目标四卡的进程，则停止本计划，不终止该进程。读写路径、权限或采集工具不满足时保存诊断记录后修正，不盲目重启。

远端只读命令（由本地单个 SSH 连接执行并重定向到本轮 `preflight/`）：

```bash
cd $HOME/infra-deepseek-v4
cat results/active-server.json
curl --fail --max-time 10 http://127.0.0.1:30000/health
curl --fail --max-time 10 http://127.0.0.1:30000/get_server_info
nvidia-smi -i 0,1,2,3 --query-gpu=index,uuid,name,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm --format=csv
.venv/bin/python -m sglang.benchmark.serving --help
.venv/bin/python -m unittest discover -s tests -v
nsys --version
nsys status --environment
```

预期：模型和 GPU 白名单一致、现有测试通过、健康接口状态 200。`nsys status` 若因可选 CPU tracing 权限返回限制，记录该限制并在 Task 4 用实际 CUDA 采集确认，不通过 sudo 或修改系统权限绕过。

**Gate 1:** 未通过所有安全核验时不进入停服和新实验。

### Task 2: 为持续压测补齐最小参数与验证

**Files:** Modify `scripts/bench_suite.py`, `tests/test_benchmark_result.py`; create `scripts/benchmark_worker.py`, `tests/test_benchmark_worker.py`.

**Interfaces:** 保留 `validate_result(result, requests, input_len, output_len)`；新增 `request_count(concurrency, override=None) -> int`。新增 CLI `--num-prompts`、`--warmup-requests`、`--seed`、`--output-dir`；省略时行为与旧脚本一致。`--output-dir` 下建 `results/` 和 `logs/`，不改变旧脚本默认输出位置。每个 run.json 增加实际请求数、warmup、seed、阶段标签和服务参数快照引用。

- [x] 按下面测试先观察新增函数缺失和非空错误未被拒绝的失败，再修改实现。

```python
def test_explicit_count_is_not_below_client_concurrency(self):
    self.assertTrue(hasattr(self.bench, 'request_count'))
    self.assertEqual(self.bench.request_count(8, 64), 64)
    with self.assertRaises(ValueError):
        self.bench.request_count(32, 16)

def test_success_strings_are_not_mistaken_for_errors(self):
    self.bench.validate_result(dict(self.result, errors=[''] * 4), 4, 512, 128)
    with self.assertRaises(ValueError):
        self.bench.validate_result(dict(self.result, errors=['', '', 'failed', '']), 4, 512, 128)

def test_actual_lengths_must_match(self):
    with self.assertRaises(ValueError):
        self.bench.validate_result(dict(self.result, output_lens=[127, 129, 128, 128]), 4, 512, 128)
```

运行：`python3 -m unittest discover -s tests -p test_benchmark_result.py -v`。预期新增测试先 FAIL，保留红灯输出，现有测试不能先变绿再假称 TDD。

- [x] 添加请求计数纯函数、四个 argparse 参数及 command 中对应值；验证 override 必须为正整数且不低于并发；旧默认 `max(4, concurrency*2)` 不变。

```python
def request_count(concurrency, override=None):
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError('Concurrency must be a positive integer')
    if override is None:
        return max(4, concurrency * 2)
    if not isinstance(override, int) or isinstance(override, bool) or override < concurrency:
        raise ValueError('Prompt count must be an integer not below concurrency')
    return override
```

- [x] 增强 `validate_result`：保留旧聚合校验；duration/throughput 必须有限且为正；若存在 `errors`，长度须为 requests 且所有值为空字符串；若存在 `input_lens`/`output_lens`，逐项等于固定目标。不要求旧测试夹具一定具有新增明细字段。
- [x] TDD 补充 CLI 默认回归、唯一 tag 拒绝覆盖、输出目录越界拒绝、seed 和 warmup 传递测试；全量测试通过后同步这两个文件，在远端重跑测试，保存脚本 SHA-256。
- [x] pilot 与正式数据用不同 `--output-dir`，同一个正式 case 的普通/MTP 使用相同 seed 和同一输入请求清单。记录实际输入清单 hash；若原生 random 数据生成不能稳定复现，先修复请求清单保存/回放，不用“同 seed”替代已验证的配对输入。

**Gate 2:** 新旧本地测试与远端测试通过；明确 raw JSONL 每个请求的有效输出长度及错误口径。

### Task 3: 取得性能边界数据（不启用 profiler）

**Files:** Modify/create `scripts/mtp_experiment.py`, `tests/test_mtp_experiment.py`; create 本轮 `pilot/`、`baseline/`、`gpu/`、`pairing.json`；preserve old results。

**Interfaces:** Task 2 CLI 输出一行一个正式 JSONL 与 run.json。配对键固定为 workload、concurrency、round、seed、request_count、input-manifest hash；mode 不属于配对键，而是对照变量。

- [x] 建立 3 workloads × 3 concurrencies × 2 modes × 3 rounds，共 **54 个初始正式 case**：short=512/128、long-context=2048/256、long-decode=512/512；c=1/8/32；none/固定3步MTP。先做 short c=1 的两种模式，确认安全停启和结果采集工作，再展开。
- [x] 对每个 workload/c 的配对运行 pilot，起始请求数 `max(32,5*c)`；用较快配置测得的请求率计算 `ceil(30*max_request_rate)` 与起始数的较大值，冻结为两个配置共同的正式请求数。正式仍不足约30秒时两个配置一起重跑到新目录；旧短轮次保留且标为非验收数据。
- [x] 交替轮次的模式顺序为 round1 none→MTP、round2 MTP→none、round3 none→MTP；轮次作为独立单位。case 内预热完成后才计时，记录 warmup 与 acceptance 计数边界。
- [x] 每轮检查 idle/health、开始与结束的 GPU 快照和每秒四卡遥测；正式 benchmark 的完成请求数、逐请求长度和 errors 全部校验。OOM、健康异常或长度不符时停止当前模式，不吞掉异常继续生成均值。
- [x] 若 short c=8/32 收益转折，则同负载补 c=16；输入长度归因增加 2048/128 桥接对照；低并发边界按结果补 c=2/4，不默认跑全笛卡尔积。

一轮命令示例（本轮目录变量由 manifest 唯一确定；这是执行期命令，不在计划阶段运行）：

```bash
cd $HOME/infra-deepseek-v4
.venv/bin/python scripts/bench_suite.py   --tag short-none-r1 --input-len 512 --output-len 128   --concurrency 1 --num-prompts 64 --warmup-requests 2 --seed 42   --output-dir "$RUN_DIR/baseline"
```

`64` 仅是命令演示：执行时替换为该 case 冻结的 pilot 请求数，不把该示例当作满足30秒的实测结果。每个 tag 必须包含 workload、mode、round；多次重试另加 attempt 标签。

安全停启：先核验 Task 1 的运行状态，再通过 `.venv/bin/python scripts/stop.py` 停止当前项目 server；确认父进程与四个 worker 都已退出、端口释放和目标四卡内存回落。以现有 `launch.py` 启动模式，记录实际 Popen PID 到 `active-server.json`，等待 health 成功后核对 `/get_server_info`，不要只记录外层 shell PID。启动记录必须使用临时文件原子替换；清理只针对该次记录的进程。

**Gate 3:** 54 case 均有效且具有唯一配对、请求清单、持续时间和模式参数记录；无法运行项清楚列出，不能用空值或零替代。

### Task 4: 六个代表配置的阶段时间线

**Files:** Create 本轮 `profiles/`、`profile-manifest.json` 和 `profile-diagnostics.log`。

**Interfaces:** 固定工作负载分别是 short c=1、short c=32、long-context c=8，各含 none/MTP。输出每个配置四个 TP rank 的有效 GPU 事件、阶段标记、采集命令、窗口和校验结果。

- [x] 先只用 short c=1 MTP 做5个forward左右的工具冒烟；使用安装版 SGLang 的 profiler API 或 `nsys` 采集服务及其子进程，不对压测客户端采集后声称看到了GPU服务工作。
- [x] 预热后请求 `/start_profile`，发固定负载，在 finally 中调用 `/stop_profile` 并等待文件 flush。检查 HTTP 状态及 JSON success，不能只看200；记录CPU/GPU事件、真实rank和profile前后无 profiler 的耗时。

安装源码确认支持后使用的请求体：

```json
{
  "output_dir": "$HOME/infra-deepseek-v4/results/mtp-study-2026-10-30/profile-smoke",
  "num_steps": 5,
  "activities": ["CPU", "GPU"],
  "profile_by_stage": true,
  "with_stack": false,
  "record_shapes": false,
  "profile_prefix": "short-mtp-c1"
}
```

执行时 output_dir 改为本轮独立目录；一次只开启一种 profiler，避免 CUPTI 冲突。prefill 步数不足时也必须显式 stop 并 flush。若 profiler 自动停止，明确记录已经完成，不把 stop 的“未运行”响应误认为当前健康失败。

- [x] 检查 trace 能解压、能解析、各TP rank非空、有CUDA kernel活动；查找 `draft`、`verify`、`draft_extend` 及correlation。只有CPU标签或无GPU事件即判该次采集无效，保存失败诊断。
- [x] 需要Nsight补足通信时，独立重启为 `nsys profile --trace=cuda,nvtx,osrt` 包裹原服务命令，再以安装版支持的 capture-range 限定热身后窗口；实际参数先由 `nsys profile --help` 核对，采集后运行 `nsys stats` 并保存原文件。禁止把PyTorch profiler与Nsight同时采集。
- [x] 冒烟通过后扩展到六个代表配置；CUDA Graph 设置与普通基准一致。缺少阶段映射时先审计现有标记，确需诊断性关闭graph则另立case，不能替换原配置证据。

**Gate 4:** 六个配置均有有效trace；工具限制与缺失rank单独列出。软件“存在”不能代替采集成功。

### Task 5: 离线统计、接受长度审计和瓶颈解释

**Files:** Create `scripts/mtp_analysis.py`, `tests/test_mtp_analysis.py`；create 本轮 `analysis/summary.json`、`analysis/paired-runs.csv`、`analysis/attribution.json`、`analysis/decision.md`。

**Interfaces:** `interval_union_us(intervals) -> float` 输入同一rank同一时钟的 `(start_us,end_us)`；`paired_ratios(records) -> list[float]` 输入已配对的正式none/MTP记录；CLI `python3 scripts/mtp_analysis.py --run-dir PATH --output-dir PATH` 离线读取，不连接远端。

- [x] 先添加并运行重叠区间测试，确认缺少分析函数时失败；再实现区间并集。

```python
def test_overlapping_communication_is_not_added_twice(self):
    self.assertEqual(self.analysis.interval_union_us([(0, 10), (5, 15)]), 15)
    self.assertEqual(self.analysis.interval_union_us([]), 0)
```

```python
def interval_union_us(intervals):
    ordered = sorted(intervals)
    total = 0.0
    end = None
    for start, stop in ordered:
        if stop < start:
            raise ValueError('Interval end precedes start')
        if end is None or start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total
```

- [x] TDD补充配对缺失、重复tag、不同请求数/seed/hash、pilot混入正式轮次、非有限数值、负duration、累计平均不能直接求差的拒绝测试。`paired_ratios` 每个独立轮次计算MTP tok/s除以none tok/s，再输出逐轮值、均值/中位数和range；标出三轮小样本，不用合并请求伪造独立重复。
- [x] 从Task1审计确定accept计数的生命周期、是否包含bonus token和预热；能取raw counters才计算计时区间差，优先逐请求metadata；只有累计平均时标为 `service_cumulative`，不用于证明跨case接受率稳定。
- [x] 每个rank分别报告阶段wall time、GPU busy并集、通信活动并集、计算通信交集和unknown；通信包含在draft/verify内时不能再次相加。无关键路径证据时不把全部通信时间写成性能损失；GPU idle不直接等于调度瓶颈。
- [x] `decision.md` 按“直接测量/对照推断/未归因”给出主假设证据，选择一个候选：现有adaptive配置、固定深度修正，或有数据支持的其他瓶颈；不预先认定接受率不变、通信主导或算法创新。

运行：`python3 -m unittest discover -s tests -p test_mtp_analysis.py -v`，再全量测试。预期分析测试先红后绿，原有测试继续通过。

**Gate 5:** 根据证据锁定一个改进检验；若主假设不成立或无法归因，记录原因并缩小结论，不跳过报告。

### Task 6: 固定深度筛选、冻结一个候选并独立复验

**Files:** Modify `scripts/launch.py`, `tests/test_launch.py`；create 本轮 `tuning/`、`validation/`、`frozen-policy.json`、`policy-decisions.jsonl`。

**Interfaces:** `build_launch(root, gpu_ids, speculative, port, gpu_addressing='uuid', speculative_steps=3, adaptive_config=None)`；CLI增加 `--speculative-steps`、`--adaptive-config`；只允许当前研究的1/2/3步。默认命令必须与旧版本相同，GPU UUID与loopback规则不变；adaptive_config必须是存在的项目内JSON文件，并先通过已安装解析器校验。

- [x] 先写失败测试：MTP1/2/3步分别映射验证容量2/3/4；普通模式拒绝非默认MTP参数；非法步数与路径越界拒绝；默认命令及白名单完全保持原样。

```python
def test_draft_capacity_tracks_topk_one_depth(self):
    for steps in (1, 2, 3):
        command, env = self.launch.build_launch(
            ROOT, [0, 1, 2, 3], 'mtp', 30000,
            speculative_steps=steps)
        self.assertEqual(command[command.index('--speculative-num-steps') + 1], str(steps))
        self.assertEqual(command[command.index('--speculative-num-draft-tokens') + 1], str(steps + 1))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], ','.join(self.launch.ALLOWED_GPUS[i] for i in (0,1,2,3)))
```

- [x] 更新固定深度分支，仍保持top-k=1。已有MTP分支中只用下列容量关系替换固定数值；其他后端与显存参数不动。

```python
if speculative_steps not in (1, 2, 3):
    raise ValueError('Study supports only 1, 2, or 3 speculative steps')
if speculative == 'mtp':
    command.extend([
        '--speculative-algorithm', 'EAGLE',
        '--speculative-num-steps', str(speculative_steps),
        '--speculative-eagle-topk', '1',
        '--speculative-num-draft-tokens', str(speculative_steps + 1),
    ])
```

- [x] adaptive分支只传安装版真实支持的开关与config路径；Task1已保存解析器与配置schema，不在计划中猜JSON字段。保存能被该解析器接受的完整配置与hash，候选集合限定1/2/3；安装不支持就记录固定深度对照，不改框架核心以掩盖不支持。
- [x] 对Task5选定的代表点测试固定1/2/3步，使用tuning数据选一个候选。冻结batch映射、请求清单split、seed和成功门槛；真实batch来自运行日志而非直接等同客户端concurrency。zero-step不作为普通解码，也不加入默认候选。
- [x] 用未参与选策略的新请求/seed，复验普通解码、原3步、该场景最佳固定深度和冻结自适应。每组至少3轮，约5%收益至少5轮；对tail决定的取舍另收集至少1000个请求。
- [x] 在冻结配置上用1→32→8的变负载回放确认步数是否真实切换；保存timestamp、actual batch、active steps、切换次数和开销。请求到达与completion日志单独保存；按总实际输出tokens/总回放wall time计算合计吞吐，不平均三个阶段tok/s。
- [x] 验收：相对原3步MTP约5%以上且超过独立重复波动；TTFT和P99相对原3步各不恶化超过5%，同时披露对最佳固定与普通解码的结果。新增错误、token长度异常或正确性冒烟失败即不保留；不保证固定seed下所有token完全相同，不把简单冒烟等同完整质量评估。

**Gate 6:** 给出支持/不支持候选的独立证据；负结果也是有效交付，不以升级依赖或改变模型来制造改善。

### Task 7: 回收、报告、恢复服务和最终验证

**Files:** Create `REPORT-MTP-BOUNDARY.md`、本轮 `SHA256SUMS`、`completion.json`；append `README.md` 研究入口；update本计划完成状态。

**Interfaces:** 只消费Task3/4/5/6原始证据；最终报告每个关键结论指向文件、case与rank/窗口。`completion.json` 包含所有case状态、未完成项、最终服务参数、health结果、测试结果和归档hash。

- [x] 远端以项目目录范围归档本轮结果与日志；本地解包前拒绝绝对路径、`..`、链接与特殊设备，校验SHA-256逐文件一致；不传模型权重或密钥。活动日志先取明确快照，避免边写边hash导致不一致。
- [x] 写报告：范围和版本→配对测量方法→吞吐与尾延迟边界→接受长度口径→每rank阶段与通信重叠→主假设支持程度→冻结候选与独立复验→负结果/限制→复现命令与证据索引。没有数据不提前生成数值或宣称本阶段完成。
- [x] 用本轮记录安全停止诊断服务，恢复Task1保存的稳定四卡MTP配置；若候选确有收益但尚未批准常驻，仍恢复原稳定配置。确认新PID、UID、cmdline、四卡UUID、/health、/v1/models和一次中文/算术生成，更新active-server记录为实际PID。
- [x] 在本地与远端运行全部tests；逐一检查54初始case、桥接或明确限制、六个代表trace、独立改进验证、无重复/缺失配对和真实token计数。只有全部成功才写 `completed`；否则为 `partial` 并列出blocked项。
- [x] README只追加新报告与复现入口，明确旧REPORT为部署阶段记录；不把旧observed-console指标当作本轮数据。更新本计划复选框和检查点hash，当前非Git仓库不执行commit。

```bash
cd $LOCAL_HOME/Documents/infra
python3 -m unittest discover -s tests -v
python3 scripts/mtp_analysis.py --run-dir "$LOCAL_RUN_DIR" --output-dir "$LOCAL_RUN_DIR/analysis-final"
```

预期：测试0失败；analysis CLI拒绝不完整配对而非填零；报告各结论均可沿证据索引回查。最终只报告实际验证通过的结果。

## 设计覆盖与阶段检查点

| spec要求 | 任务 | 可核验产物 |
|---|---|---|
| 固定模型、GPU与版本、安全停启 | 1、3、7 | manifest、命令日志、恢复health |
| 三类负载与并发边界 | 2、3 | 正式JSONL、pilot、pairing.json |
| 输入长度单因素桥接 | 3 | 2048/128与512/128配对，或明确限制 |
| 阶段profile与工具扰动 | 4 | 六个配置四rank有效trace、采集日志 |
| 接受长度、通信重叠与未知项 | 1、5 | 源码审计、attribution.json |
| 一个改进假设与负结果 | 5、6 | decision.md、tuning、validation |
| 冻结策略与独立样本 | 6 | frozen-policy.json、split/seed/hash |
| 可复现报告与原始证据 | 7 | REPORT-MTP-BOUNDARY.md、SHA256SUMS |

执行顺序：Task1→Task2→Task3→Task4→Task5→Task6→Task7。Task3首个配对点、Task4采集冒烟、Task5证据判断后分别汇报一次，不在每个微小步骤重复征求已获批准的范围。

## 执行记录

- 2026-10-30：Task 1 已完成，证据目录 `results/mtp-study-2026-10-30/run-20261030T070100Z`；GPU安全核验、原服务恢复命令和原始源码已固定。Nsight CPU采样受限，CUDA/NVTX采集在Task4冒烟确认。
- Task 2 已完成：持续压测参数、精确 token 请求清单、请求级指标和本地/远端 25 项测试均已验证；短冒烟结果已回收。
- Task 3 进行中：先执行单个正式 pilot，再展开配对基准。

- Task 3：18 个独立 pilot 已完成，54-case baseline 后台执行器 PID 90680；遥测和事件逐条写入本轮目录。Task 4/5/6 的纯本地辅助函数已预先通过测试，未干扰远端基准。

- 预检补充：共享verify函数没有整个函数的外层标记，但模型执行器已有 `step[VERIFY bs=…]`。Task4使用该标记归因 `verify_forward`，不冒充接受判断与KV整理的完整verify开销；不为加标记修改已安装框架。
- 本地准备：47项测试通过；新的 profiling、analysis 与 tuning 辅助代码尚未同步，保持运行中的baseline代码固定。验证提示集hash `2cd295a606d1857aff751037d153395cba829fc145a4ea372bb000f4e3ce4ba8`。

- Task3 已完成：54正式case、27配对、7032请求；并发16与2048/128桥接共12case完成，输入桥接的200份token前缀一致性已核验。
- Task4 已完成：六个配置×四rank的GPU trace均有效；原始文件已回收到本地profiles-remote并通过130文件SHA-256校验。
- Task5 已完成：排除了CUDA Graph GPU注释重复计数后重算24rank；证据决策保存在analysis/decision.json，允许高batch固定深度筛选。
- Task6 进行中：固定1/2/3步无profiler筛选；候选冻结后执行独立提示集五轮验证。

- Task6 已完成：9组固定深度筛选、冻结映射、独立提示集4配置×5轮（10,240请求）、4配置×3段回放及20项正确性冒烟均已核验。固定1步和冻结映射的吞吐提高约7.6%与7.4%，TTFT约为原3步的2.65倍，依预设门槛不保留候选。
- 回放验收边界：保存时间戳和实际状态切换日志，报告总wall及benchmark窗口吞吐。逐次切换的精确GPU关键路径开销未能独立隔离，作为明确限制保留，不用日志消息间隔冒充GPU计时。该项不作为候选收益的独立论据。
- Task7 已完成：107组、26,776个正式请求、24rank trace及1,200次桥接请求前缀校验通过；四类快照的962条文件哈希记录复核。本地／远端各50项测试通过，原稳定TP4固定3步MTP已恢复，PID 235501。报告、README、原始快照、服务验收和completion记录均已更新；逐文件摘要为本轮SHA256SUMS。

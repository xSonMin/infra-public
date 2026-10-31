> 公开脱敏副本；环境标识已替换，原始证据另行保留。详见仓库根目录 PUBLICATION.md。

# DeepSeek V4 Flash / RTX PRO 6000 实测记录

日期：2026-10-30。状态：已完成真实推理、补测、原始结果回收和最终服务验收。

## 已确认的部署与范围

- 远端工作目录：`$HOME/infra-deepseek-v4`，用户 `research-user`，主机名 `research-user`。
- 模型：`deepseek-ai/DeepSeek-V4-Flash`；revision `60d8d70770c6776ff598c94bb586a859a38244f1`。
- 模型 46 个 safetensors 分片已完整下载，官方文件总大小约 159.6 GB。V4-Pro 权重约 864.7 GB，超出获准四卡的总显存，因此本轮选用 Flash。
- 物理 GPU 0、1、2、3：NVIDIA RTX PRO 6000 Blackwell Server Edition，各 97887 MiB，SM120，驱动 580.178.04。实验只使用这四张卡。
- 用户环境：Python 3.12.16、SGLang 0.5.23、PyTorch 2.14.1+cu130、FlashInfer 0.7.2、Transformers 5.19.0。依赖安装及编译均在用户目录完成。
- 预检通过：四卡可见、SM120、四卡 P2P 矩阵全为 `True`，Triton 编译与执行通过。
- 启动器验证物理卡 UUID 和实际 CUDA 可见设备顺序；默认使用 UUID 限定设备。双卡自定义通信实验使用经过 UUID 校验的 ordinal 形式。

## 压测条件

固定输入 512 token、输出 128 token、种子 42；使用 `sources/benchmark-prompts.json` 的受控中文提示集。原生 SGLang serving benchmark 通过整数 token ID 精确控制输入长度，关闭前缀缓存，每个并发度单独预热。并发 1、8、32 分别完成 4、16、64 个计时请求。吞吐为实际输出 token 数除以计时区间；TTFT 为首 token 延迟，TPOT 为扣除首 token 后的逐 token 平均耗时。

基线与 MTP 使用同一模型、后端及负载。MTP 使用模型自带头，EAGLE 路径、3 步、top-k=1、4 个 draft token。四卡普通解码和四卡 MTP 均有三轮原始 JSONL；双卡筛选各配置完成一轮。所有原始文件已回收至 `results/remote-2026-10-30/results/`。

## 三轮四卡平均结果

| 并发 | TP4 普通解码 输出 tok/s | TP4 MTP 输出 tok/s | MTP/普通 | 普通 TTFT ms | MTP TTFT ms | 普通 TPOT ms | MTP TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 101.22 | 188.39 | 1.861x | 87.69 | 93.95 | 9.252 | 4.572 |
| 8 | 588.83 | 758.18 | 1.288x | 316.83 | 323.15 | 11.159 | 7.983 |
| 32 | 1288.76 | 1321.95 | 1.026x | 815.98 | 679.02 | 18.551 | 18.654 |

四卡 MTP 在单请求时吞吐约为普通解码的 1.86 倍，并发 8 时约为 1.29 倍；并发 32 时只有约 2.6% 的吞吐提升。并发 32 的 MTP 平均 TTFT 较低，但 TPOT 基本相同，因此不能把低并发收益外推到高并发。

三轮原始吞吐范围如下：普通解码在并发 1、8、32 分别为 101.15–101.28、588.45–589.25、1286.77–1292.00 tok/s；MTP 分别为 188.08–188.89、739.99–767.55、1317.41–1328.55 tok/s。

较长输入补测使用输入 2048、输出 256、并发 8，16 个请求均成功：

| 配置 | 输出 tok/s | 平均 TTFT ms | 平均 TPOT ms | MTP/普通 |
|---|---:|---:|---:|---:|
| TP4 普通解码 | 505.81 | 777.24 | 12.813 | — |
| TP4 MTP | 580.15 | 651.22 | 10.529 | 1.147x |

## 双卡筛选

| 配置 | 并发 1 输出 tok/s | 并发 8 输出 tok/s | 并发 32 输出 tok/s |
|---|---:|---:|---:|
| TP2 普通解码 / NCCL | 90.33 | 516.62 | 1171.13 |
| TP2 普通解码 / custom all-reduce | 93.85 | 532.20 | 1205.45 |
| TP2 MTP / custom all-reduce | 165.20 | 582.05 | 1179.99 |

双卡单副本在本负载下没有超过四卡 MTP 的绝对吞吐。两个双卡副本的总服务吞吐未做实测，不把单副本结果乘二作为结论。

## 正确性和最终服务验收

- OpenAI 兼容 chat 接口成功生成中文介绍并正确回答 `17×19=323`。
- MTP 接口通过 `17×19=323`、`1 到 100 的整数之和=5050`、`3 的三次方=27` 三个独立检查。
- 29 组原始压测 JSONL 均满足成功请求数、输入 token 总数、输出 token 总数和吞吐公式校验；所有请求错误列表为空。
- 本地恢复后的 15 项测试全部通过；新增的 `scripts/stop.py` 和 `tests/test_stop.py` 已同步到远端，远端同样通过 15 项测试。
- 后台工作流写出 `results/final-experiments-complete.json`，最终选择 `final-tp4-mtp`，选择指标为“包含 prefill 的单请求输出吞吐”。
- 最终服务记录为 PID `35780`，监听 `127.0.0.1:30000`，模型名 `deepseek-v4-flash`，`/health` 返回 HTTP 200，`/v1/models` 返回该模型。
- 最终服务命令使用 TP=4、`flashinfer_mxfp4`、EAGLE 3 步、top-k=1、4 个 draft token，并通过四个获准 GPU UUID 设置 `CUDA_VISIBLE_DEVICES`。验收时四张卡各占用约 90661 MiB / 97887 MiB。

这些检查属于服务可用性和简单输出正确性，不等同于完整模型质量评测。吞吐结论针对本提示集、固定输入输出长度、关闭前缀缓存的稳态测量；预热和首次编译时间不计入吞吐。

## 远端连接恢复与交付物

本轮 SSH 已恢复。之前的 `kex_exchange_identification: Connection closed by remote host` 未再复现；恢复后先检查了工作流、服务、进程和显存，再回收原始文件，没有重复启动最终服务。

本地原始快照：

- `results/remote-2026-10-30/`：远端 `results/`、`logs/`、脚本、测试和提示集的恢复快照，共 139 个文件。
- `results/remote-2026-10-30.tar.gz`：上述快照的压缩归档，SHA-256 为 `0e6f917acffd60a28b109a22d5f5cce520288393f2844baaff27355b754a4a38`。
- `results/final-summary.json`：从原始 JSONL 重新计算的汇总和验收记录。
- `results/recovery-remote-tests.log`：远端新增停止脚本后的 15 项测试输出。
- `results/observed-console-metrics.json`：断线前的控制台转录，保留作为历史记录；最终数值以回收的原始 JSONL 为准。

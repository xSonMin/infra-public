> 公开脱敏副本；环境标识已替换，原始证据另行保留。详见仓库根目录 PUBLICATION.md。

# DeepSeek V4 Flash 在四张 RTX PRO 6000 上的推理加速

用户授权：只使用远端物理 GPU 0、1、2、3。目标主机为 `research-user@gpu-node`。

## 范围和实测资源

本次对象选择 `deepseek-ai/DeepSeek-V4-Flash`，revision `60d8d70770c6776ff598c94bb586a859a38244f1`。这是机器名称对应的 Flash 版本；不把 V4-Pro 或 V4.1 的结果混入。官方 Flash 权重文件共 159617149040 字节；Pro 权重 864721029744 字节，四张卡无法全显存容纳。

GPU 0–3 均为 RTX PRO 6000 Blackwell Server Edition，单卡 97887 MiB，驱动 580.178.04，CUDA toolkit 13.0，初始显存占用为 0 MiB。当前账户没有推理依赖或模型缓存。用户级安装，不改系统驱动、不终止其他用户进程。

## 方案和取舍

1. 四卡 TP=4：作为首次可运行基线，显存余量充足；PCIe 通信可能影响单请求解码。
2. 双卡 TP=2：减少每步通信，预期适合低并发；模型和 KV cache 可用余量较少。
3. 两个 TP=2 副本：使用全部四卡服务独立请求，比较总体吞吐；仅在单副本验证后评估。

优先使用 SGLang、原始 FP4 专家权重、`flashinfer_mxfp4` 后端。先确认 CUDA 和 SM120 内核实际可用，再比较 MTP 推测解码。保持模型、输入和输出长度、随机种子、请求数一致；不以换小模型、缩短输出或只测缓存命中伪装加速。

## 交付和完成标准

交付可重复执行的启动/停止及压测命令、固定依赖与权重 revision、原始结果、中文实测报告。至少成功完成真实推理，并用同一工作负载比较基线和候选方案；以实际输出 token 计算吞吐，区分 TTFT、TPOT、总输出吞吐及每卡吞吐。未完成或失败实验单独记录。默认仅监听 127.0.0.1。所有推理进程通过 UUID 限定到授权四卡。

## 资料核验

2026-10-30 实时读取 Hugging Face 官方模型元数据与 SGLang 官方配方。SGLang main revision `80afdc1280bf37c1440e6e7009741cbc94e2fba0` 的配置将原始 Flash/RTX PRO 6000 TP=2 标为 verified=true，0731 Flash Official 对应项为 verified=false；不能混淆两项。运行时支持仍需实测。PyPI 当时 SGLang 稳定版为 0.5.23，具体兼容性待安装验证。

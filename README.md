# verl Stage Tuning Agent

这个目录实现了一个参考 OptiCo 的 verl 0.7 GRPO 自动调优闭环：Proposal Agent 生成参数，确定性 Validator 检查硬约束，Feasibility Agent 审查多阶段显存 trade-off，Diagnosis Agent 在失败后归因。对外使用 shell 脚本，Python 仅作为内部实现。

## 调优状态

1. `hardware_tuning` / `hardware_repair`
   - 每个候选最多运行 20 update。
   - 前 5 update 是 Resource Gate；出现 OOM/NCCL 或 GPU 显存达到硬上限时提前停止。
   - 通过后继续测量去掉 warmup 后的端到端吞吐和四个训练子阶段耗时。
2. `stability_tuning`
   - 默认运行 80 update。
   - 冻结硬件参数，只允许修改 lr、warmup、KL、entropy 和 rollout.n。
3. `confirm`
   - 默认运行 300 update。
   - 配置冻结，记录 reward 到阈值的累计时间、累计 token 和 peak reward。

一次 update 内的监控阶段为 `rollout`、`actor_log_prob`、`ref_log_prob`、`training`。runner 设置 `VERL_LOGGING_LEVEL=DEBUG`，使用 verl 的 `GPUMemoryLogger` 阶段边界，同时调用平台对应的 SMI 每秒采样每张卡。若没有可用 SMI，阶段显存会使用日志观测值；没有可靠数据时输出 `null`。

支持的平台：

- `PLATFORM=V5000`：默认平台，使用 `xpu-smi` 和 `train/env_V5000.sh`。
- `PLATFORM=A100`、`NVIDIA` 或 `CUDA`：使用 `nvidia-smi` 和 `train/env_NVIDIA.sh`。
- `PLATFORM=C550` 或 `METAX`：使用兼容的 `nvidia-smi` 和 `train/env_C550.sh`。
- 特殊环境可通过 `GPU_SMI` 和 `VERL_ENV_SCRIPT` 覆盖监控命令与环境脚本。

## 配置

日常修改集中在两个 JSON 文件：

- `config/base_parameters.json`：由参考脚本 `qwen3_8B_baseline.sh` 转换出的初始 verl Hydra 参数。数据集、模型路径、GPU 数在这里修改。
- `config/agent_config.json`：verl 仓库路径、update 预算、显存阈值和调优轮数。

服务器上的 verl 路径可以直接通过 `VERL_ROOT` 设置。环境脚本只能设置环境变量，不能包含训练命令。API 凭据通过环境变量提供：

```bash
export VERL_ROOT=/workspace/verl
export API_KEY=...
export BASE_URL=...
export INFER_MODEL=...
```

## 运行

先检查第一轮生成的命令，不启动训练：

```bash
PLATFORM=V5000 bash agents/verl_agent/run_circle.sh --dry-run --rules-only
```

运行一个 trial。第一轮是 20-update 硬件基线，不需要调用 LLM：

```bash
PLATFORM=V5000 MAX_TRIALS=1 bash agents/verl_agent/run_circle.sh
```

后续运行会读取 `output/verl_agent/trials.jsonl`，调用 Agent 产生候选并继续状态机：

```bash
PLATFORM=V5000 MAX_TRIALS=10 bash agents/verl_agent/run_circle.sh
```

默认一次只运行一个 trial，便于检查实际 GPU 环境。确认配置和监控正确后再提高 `--max-trials`。

主要输出：

- `agents/verl_agent/output/trials.jsonl`：所有 trial 的参数、指标、建议和 feasibility 结果。
- `agents/verl_agent/output/state.json`：当前调优阶段。
- `agents/verl_agent/output/trials/NNNN/train.log`：原始 verl 日志。
- `agents/verl_agent/output/trials/NNNN/gpu_samples.csv`：带训练子阶段标签的逐 GPU 采样。
- `agents/verl_agent/output/trials/NNNN/trial_report.json`：单轮结构化报告。

## 最终指标

`tools/compare_end_to_end_reward.py` 是独立验收入口：

```bash
PYTHONPATH=agents python3 agents/verl_agent/tools/compare_end_to_end_reward.py \
  --log base=path/to/base.log \
  --log candidate=path/to/candidate.log \
  --output reward_comparison.json
```

阈值与窗口可以通过 `--thresholds` 和 `--window` 修改。

## 独立脚本

主循环之外，也可以单独执行与 OptiCo 对应的步骤：

```bash
# 单独调用一个 Agent 角色
bash agents/verl_agent/analyzer/run_analyze.sh proposal context.json suggestion.json

# 单独分析已有训练日志
bash agents/verl_agent/monitor/run_monitor.sh train.log gpu_samples.csv report.json

# 单独运行一个受监控的 trial
PLATFORM=V5000 bash agents/verl_agent/train/run_verl.sh \
  parameters.json hardware_tuning 1 20
```

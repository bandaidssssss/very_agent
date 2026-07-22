#!/usr/bin/env python3
"""
提取实验目录中每个 trial 的 agent 行为（工具调用、参数变更、决策原因）并生成 Markdown 报告。

用法:
    python3 extract_agent_trace.py ssh_agent/output/0720_1656_2026 [output.md]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def _compact_json(obj, max_len=120):
    """紧凑的 JSON 字符串，超长截断。"""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def extract_tool_calls(tool_calls: list[dict]) -> str:
    """格式化工具调用列表为 Markdown 表格。"""
    if not tool_calls:
        return "_无工具调用_\n"

    lines = [
        "| 轮次 | 工具 | 参数 / 查询内容 | 状态 |",
        "|---|---|---|---|",
    ]
    # Group by tool_round
    by_round: dict[int, list[dict]] = {}
    for tc in tool_calls:
        by_round.setdefault(tc.get("tool_round", 0), []).append(tc)

    for rnd in sorted(by_round):
        for tc in by_round[rnd]:
            name = tc.get("name", "?")
            args = tc.get("arguments", {})
            status = tc.get("status", "?")
            # 简化参数展示
            if name == "parameter_understanding":
                items = args.get("items", [])
                arg_summary = ", ".join(
                    [it.split(".")[-1] for it in items[:6]]
                )
                if len(items) > 6:
                    arg_summary += f" …共 {len(items)} 个"
            elif name == "memory_estimator":
                changes = args.get("changes", {})
                ref = args.get("reference_trial_id", "?")
                param_names = ", ".join(
                    [k.split(".")[-1] for k in changes.keys()]
                )
                arg_summary = f"ref_trial={ref}, 预测参数: {param_names}"
            elif name == "tuning_strategies":
                items = args.get("items", [])
                arg_summary = ", ".join(items)
            elif name == "query_trial_history":
                arg_summary = f"stage={args.get('stage')}, limit={args.get('limit')}"
            elif name == "search_verl_docs":
                arg_summary = f'查询: "{args.get("query", "")}"'
            else:
                arg_summary = _compact_json(args, 100)
            lines.append(f"| {rnd} | `{name}` | {arg_summary} | {status} |")
    return "\n".join(lines) + "\n"


def extract_proposal_changes(proposal: dict | None) -> str:
    """格式化 proposal 中的参数变更为 Markdown 表格。"""
    if not proposal:
        return "_无 proposal_\n"

    decision = proposal.get("decision", "?")
    reason = proposal.get("reason", "")
    confidence = proposal.get("confidence")
    changes = proposal.get("changes", {})

    lines = [
        f"- **决策**: `{decision}`",
        f"- **原因**: {reason}",
    ]
    if confidence is not None:
        lines.append(f"- **置信度**: {confidence}")
    lines.append("")

    if changes and isinstance(changes, dict) and decision == "modify":
        # 判断 changes 是否是 {key: {from, to, reason}} 结构
        first_val = next(iter(changes.values()), None)
        if isinstance(first_val, dict) and "from" in first_val:
            lines.append("| 参数 | 旧值 | 新值 | 原因 |")
            lines.append("|---|---|---|---|")
            for key, ch in changes.items():
                short_key = key.split(".")[-1]
                lines.append(
                    f"| `{short_key}` | `{ch.get('from')}` | `{ch.get('to')}` "
                    f"| {ch.get('reason', '')} |"
                )
        else:
            # 简易格式 {key: value}
            lines.append("| 参数 | 目标值 |")
            lines.append("|---|---|")
            for key, val in changes.items():
                short_key = key.split(".")[-1]
                lines.append(f"| `{short_key}` | `{val}` |")
        lines.append("")

    return "\n".join(lines)


def extract_feasibility(feas: list[dict] | None) -> str:
    """格式化 feasibility reviews。"""
    if not feas:
        return ""
    lines = ["### Feasibility 审查", ""]
    for i, f in enumerate(feas):
        verdict = f.get("verdict", "?")
        reason = f.get("reason", "")
        lines.append(f"- **#{i+1} 判定**: `{verdict}` — {reason}")
    lines.append("")
    return "\n".join(lines)


def extract_rejections(rejections: list[dict] | None) -> str:
    """格式化 rejections。"""
    if not rejections:
        return ""
    lines = ["### 建议被拒绝记录", ""]
    for i, r in enumerate(rejections):
        source = r.get("source", "?")
        violations = r.get("violations", [])
        lines.append(f"- **#{i+1}** 来源: `{source}`")
        for v in violations:
            lines.append(f"  - ❌ {v}")
    lines.append("")
    return "\n".join(lines)


def _fmt(val, spec=".1f"):
    """安全格式化数值，处理 None。"""
    if val is None:
        return "-"
    return format(val, spec)


def extract_metrics(trial: dict) -> str:
    """提取关键指标。"""
    lines = ["### 关键指标", ""]

    perf = trial.get("performance", {})
    if perf:
        tp = perf.get("throughput", {})
        tps = perf.get("time_per_step_s", {})
        tgs = perf.get("generation_tgs", {})
        mfu = perf.get("actor_mfu", {})
        bottleneck = perf.get("time_bottleneck", "")
        lines.append("| 指标 | 均值 | P95 | 最大值 |")
        lines.append("|---|---|---|---|")
        if tp:
            lines.append(f"| 吞吐量 (tok/s) | {_fmt(tp.get('mean'), '.1f')} | {_fmt(tp.get('p95'), '.1f')} | {_fmt(tp.get('max'), '.1f')} |")
        if tps:
            lines.append(f"| 每步耗时 (s) | {_fmt(tps.get('mean'), '.1f')} | {_fmt(tps.get('p95'), '.1f')} | {_fmt(tps.get('max'), '.1f')} |")
        if tgs:
            lines.append(f"| 生成 tgs | {_fmt(tgs.get('mean'), '.1f')} | {_fmt(tgs.get('p95'), '.1f')} | {_fmt(tgs.get('max'), '.1f')} |")
        if mfu:
            lines.append(f"| Actor MFU | {_fmt(mfu.get('mean'), '.4f')} | {_fmt(mfu.get('p95'), '.4f')} | {_fmt(mfu.get('max'), '.4f')} |")
        if bottleneck:
            lines.append(f"| **时间瓶颈** | {bottleneck} |||")
        lines.append("")

        # Phase durations
        phase = perf.get("phase_duration_s", {})
        if phase:
            lines.append("**各阶段耗时 (s):**")
            lines.append("")
            lines.append("| 阶段 | 均值 | P95 | 最大 |")
            lines.append("|---|---|---|---|")
            for pname in ["rollout", "actor_log_prob", "ref_log_prob", "training"]:
                p = phase.get(pname, {})
                if p:
                    lines.append(
                        f"| {pname} | {_fmt(p.get('mean'), '.1f')} | "
                        f"{_fmt(p.get('p95'), '.1f')} | {_fmt(p.get('max'), '.1f')} |"
                    )
            lines.append("")

    stability = trial.get("stability", {})
    if stability:
        reward = stability.get("reward", {})
        kl = stability.get("actor_ppo_kl", {})
        entropy = stability.get("actor_entropy", {})
        clipfrac = stability.get("actor_pg_clipfrac", {})
        resp_len = stability.get("response_length", {})

        lines.append("**稳定性指标:**")
        lines.append("")
        lines.append("| 指标 | 均值 | P95 | 最大 |")
        lines.append("|---|---|---|---|")
        if reward:
            lines.append(f"| Reward | {_fmt(reward.get('mean'), '.4f')} | {_fmt(reward.get('p95'), '.4f')} | {_fmt(reward.get('max'), '.4f')} |")
        lines.append(f"| Reward 斜率 | {_fmt(stability.get('reward_slope'), '.6f')} |||")
        if kl:
            lines.append(f"| Actor PPO KL | {_fmt(kl.get('mean'), '.8f')} | {_fmt(kl.get('p95'), '.8f')} | {_fmt(kl.get('max'), '.8f')} |")
        if entropy:
            lines.append(f"| Actor Entropy | {_fmt(entropy.get('mean'), '.4f')} | {_fmt(entropy.get('p95'), '.4f')} | {_fmt(entropy.get('max'), '.4f')} |")
        if clipfrac:
            lines.append(f"| Clip Fraction | {_fmt(clipfrac.get('mean'), '.6f')} | {_fmt(clipfrac.get('p95'), '.6f')} | {_fmt(clipfrac.get('max'), '.6f')} |")
        if resp_len:
            lines.append(f"| Response Length | {_fmt(resp_len.get('mean'), '.1f')} | {_fmt(resp_len.get('p95'), '.1f')} | {_fmt(resp_len.get('max'), '.1f')} |")
        lines.append("")

    resource = trial.get("resource", {})
    if resource:
        lines.append(f"- **显存瓶颈**: {resource.get('memory_bottleneck', '-')}")
        lines.append(f"- **峰值显存**: {resource.get('max_observed_memory_pct', '-'):.1f}%")
        lines.append("")

    return "\n".join(lines)


def extract_param_diff(current: dict, previous: dict | None) -> str:
    """对比两个 trial 的参数差异。"""
    if not previous:
        return ""
    diffs = []
    for key in sorted(set(list(current.keys()) + list(previous.keys()))):
        cv = current.get(key)
        pv = previous.get(key)
        if cv != pv:
            sk = key.split(".")[-1]
            diffs.append(f"| `{sk}` | `{pv}` | `{cv}` |")

    if not diffs:
        return "_参数无变化_\n"

    lines = ["| 参数 | 旧值 | 新值 |", "|---|---|---|"]
    lines.extend(diffs)
    return "\n".join(lines) + "\n"


# ── 主逻辑 ──────────────────────────────────────────────

def process_experiment(exp_dir: Path, output_path: Path) -> None:
    trials_path = exp_dir / "trials.jsonl"
    if not trials_path.exists():
        print(f"错误: 找不到 {trials_path}")
        sys.exit(1)

    state_path = exp_dir / "state.json"
    state_info = ""
    if state_path.exists():
        state = json.load(state_path.open(encoding="utf-8"))
        state_info = (
            f"- **最终阶段**: `{state.get('current_stage', '?')}`\n"
            f"- **总 Trial 数**: {state.get('last_trial_id', '?')}\n"
        )

    # 读取所有 trial
    trials = []
    with trials_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))

    # 提取实验时间戳
    exp_name = exp_dir.name
    ts_match = None
    for part in exp_name.split("_"):
        if len(part) == 4 and part.isdigit():
            ts_match = part
            break

    lines: list[str] = []
    lines.append(f"# Agent 实验报告: `{exp_name}`")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**数据来源**: `{exp_dir}`")
    lines.append(f"**总 Trial 数**: {len(trials)}")
    lines.append("")
    lines.append("## 实验概览")
    lines.append("")
    lines.append(state_info)
    lines.append("")

    # 试验总览表
    lines.append("| Trial | 阶段 | 结果 | 吞吐量 | Reward (均值) | Reward (最大) | 显存峰值% | Agent 工具调用 |")
    lines.append("|---|---|---|---:|---:|---:|---:|:---:|")
    for t in trials:
        tid = t.get("trial_id", "?")
        stage = t.get("stage", "?")
        result = t.get("result", "?")
        tp = t.get("performance", {}).get("throughput", {}).get("mean", "-")
        if isinstance(tp, (int, float)):
            tp = f"{tp:.0f}"
        rw_mean = t.get("stability", {}).get("reward", {}).get("mean", "-")
        if isinstance(rw_mean, (int, float)):
            rw_mean = f"{rw_mean:.4f}"
        rw_max = t.get("stability", {}).get("reward", {}).get("max", "-")
        if isinstance(rw_max, (int, float)):
            rw_max = f"{rw_max:.4f}"
        mem = t.get("resource", {}).get("max_observed_memory_pct", "-")
        if isinstance(mem, (int, float)):
            mem = f"{mem:.1f}%"
        n_tools = len(t.get("agent_trace", {}).get("proposal_conversation", {}).get("tool_calls", []))
        tool_str = str(n_tools) if n_tools else "-"

        lines.append(
            f"| {tid} | {stage} | {result} | {tp} | {rw_mean} | {rw_max} | {mem} | {tool_str} |"
        )
    lines.append("")

    # ── 逐 Trial 详细展开 ──
    lines.append("---")
    lines.append("")
    lines.append("## 逐 Trial 详细分析")
    lines.append("")

    prev_params = None
    for idx, t in enumerate(trials):
        tid = t.get("trial_id", "?")
        stage = t.get("stage", "?")
        result = t.get("result", "?")
        updates = t.get("updates_completed", "?")

        lines.append(f"### Trial {tid}: {stage}")
        lines.append("")
        lines.append(f"- **结果**: `{result}` | **完成步数**: {updates}/{t.get('updates_target', '?')}")
        error = t.get("error", {})
        if error and error.get("type"):
            lines.append(f"- **错误类型**: {error['type']}")
        lines.append("")

        # 参数对比
        params = t.get("parameters", {})
        if prev_params is not None and params != prev_params:
            lines.append("#### 参数变更（相比上一 Trial）")
            lines.append("")
            lines.append(extract_param_diff(params, prev_params))
            lines.append("")
        elif idx == 0:
            lines.append("#### 初始参数（基准）")
            lines.append("")
            lines.append(f"_完整参数见 `trials/{tid:04d}/parameters.json`_")
            lines.append("")
        prev_params = params

        # 关键指标
        lines.append(extract_metrics(t))

        # Agent 行为
        trace = t.get("agent_trace")
        if trace:
            conv = trace.get("proposal_conversation", {})
            tool_calls = conv.get("tool_calls", [])

            lines.append("### Agent 行为分析")
            lines.append("")

            if tool_calls:
                lines.append(f"**工具调用 ({len(tool_calls)} 次):**")
                lines.append("")
                lines.append(extract_tool_calls(tool_calls))
                lines.append("")

            # Proposal 决策
            proposal = t.get("proposal", {})
            if proposal:
                lines.append("#### Agent 决策")
                lines.append("")
                lines.append(extract_proposal_changes(proposal))
                lines.append("")

            # Feasibility
            feas = trace.get("feasibility_reviews", [])
            if feas:
                lines.append(extract_feasibility(feas))

            # Rejections
            rejs = trace.get("rejections", [])
            if rejs:
                lines.append(extract_rejections(rejs))

            # 诊断
            diagnosis = trace.get("diagnosis")
            if diagnosis:
                lines.append(f"**诊断信息**: {diagnosis}")
                lines.append("")
        else:
            lines.append("### Agent 行为分析")
            lines.append("")
            if tid == 1:
                lines.append("_Trial 1 是基准试验，无需 Agent 提出变更建议。_")
            else:
                lines.append("_无 agent_trace 数据。_")
            lines.append("")

        # 实验日志路径
        log_path = t.get("log_path", "")
        if log_path:
            lines.append(f"📄 [训练日志]({log_path})")
            lines.append("")

        lines.append("---")
        lines.append("")

    # 写入文件
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 报告已生成: {output_path}")
    print(f"   共 {len(trials)} 个 trial")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp_dir = Path(sys.argv[1]).resolve()
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2]).resolve()
    else:
        output_path = exp_dir / "agent_report.md"

    process_experiment(exp_dir, output_path)


if __name__ == "__main__":
    main()

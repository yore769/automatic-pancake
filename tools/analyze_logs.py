"""
RT-DETR Training Log Analyzer

Analyzes training log files from RT-DETR experiments to compare
performance of different improvement configurations (NWD loss,
dynamic query grouping, LS convolution) and diagnose why combined
improvements may not yield additive gains.

Usage:
    python tools/analyze_logs.py --log-files log1.txt log2.txt log3.txt \
        --labels "NWD" "DynamicGroup" "Combined"

    python tools/analyze_logs.py --log-files log_combined.txt \
        --analyze-loss-components
"""

import os
import sys
import json
import argparse
import math
from collections import defaultdict


def parse_log_file(filepath):
    """Parse a JSONL-format training log file.

    Each line of the log file is expected to be a JSON object with fields
    like epoch, train_loss, train_loss_vfl, test_coco_eval_bbox, etc.

    Args:
        filepath: Path to the log file.

    Returns:
        List of dicts, one per epoch.
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping line {lineno} in {filepath}: {e}")
    return records


def extract_metrics(records):
    """Extract key training metrics from log records.

    Returns:
        dict with keys:
            epochs          - list of epoch indices
            train_loss      - total training loss per epoch
            loss_vfl        - VFL loss per epoch
            loss_bbox       - bbox L1 loss per epoch
            loss_giou       - GIoU loss per epoch
            loss_nwd        - NWD loss per epoch (if present)
            ap              - AP@0.5:0.95 per epoch
            ap50            - AP@0.5 per epoch
            ap_small        - AP_small per epoch
            ap_medium       - AP_medium per epoch
            ap_large        - AP_large per epoch
    """
    metrics = defaultdict(list)

    for r in records:
        epoch = r.get("epoch", len(metrics["epochs"]))
        metrics["epochs"].append(epoch)
        metrics["train_loss"].append(r.get("train_loss", float("nan")))
        metrics["loss_vfl"].append(r.get("train_loss_vfl", float("nan")))
        metrics["loss_bbox"].append(r.get("train_loss_bbox", float("nan")))
        metrics["loss_giou"].append(r.get("train_loss_giou", float("nan")))
        metrics["loss_nwd"].append(r.get("train_loss_nwd", float("nan")))

        bbox_eval = r.get("test_coco_eval_bbox", [])
        if len(bbox_eval) >= 12:
            metrics["ap"].append(bbox_eval[0])
            metrics["ap50"].append(bbox_eval[1])
            metrics["ap75"].append(bbox_eval[2])
            metrics["ap_small"].append(bbox_eval[6])
            metrics["ap_medium"].append(bbox_eval[7])
            metrics["ap_large"].append(bbox_eval[8])
        else:
            for key in ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large"):
                metrics[key].append(float("nan"))

    return dict(metrics)


def _safe_last(values):
    """Return the last non-nan value in a list."""
    for v in reversed(values):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return float("nan")


def _fmt(v, decimals=4):
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def print_summary(label, metrics):
    """Print a one-line summary for a single experiment."""
    ap = _safe_last(metrics["ap"])
    ap50 = _safe_last(metrics["ap50"])
    ap_s = _safe_last(metrics["ap_small"])
    ap_l = _safe_last(metrics["ap_large"])
    n_epochs = len(metrics["epochs"])
    final_loss = _safe_last(metrics["train_loss"])
    print(
        f"  {label:<30s} | epochs={n_epochs:3d} | loss={_fmt(final_loss)} "
        f"| AP={_fmt(ap)} | AP50={_fmt(ap50)} "
        f"| AP_s={_fmt(ap_s)} | AP_l={_fmt(ap_l)}"
    )


def compare_experiments(all_metrics, labels):
    """Print a comparison table for all experiments.

    Args:
        all_metrics: List of metrics dicts.
        labels:      List of experiment labels.
    """
    print("\n" + "=" * 90)
    print("实验对比汇总 / Experiment Comparison Summary")
    print("=" * 90)
    print(
        f"  {'实验标签':<30s} | {'epochs':>6s} | {'最终Loss':>10s} "
        f"| {'AP':>8s} | {'AP50':>8s} | {'AP_small':>9s} | {'AP_large':>9s}"
    )
    print("-" * 90)

    baseline_ap = None
    for label, metrics in zip(labels, all_metrics):
        ap = _safe_last(metrics["ap"])
        ap50 = _safe_last(metrics["ap50"])
        ap_s = _safe_last(metrics["ap_small"])
        ap_l = _safe_last(metrics["ap_large"])
        n_epochs = len(metrics["epochs"])
        final_loss = _safe_last(metrics["train_loss"])

        delta_str = ""
        if baseline_ap is None:
            baseline_ap = ap
        else:
            delta = ap - baseline_ap
            delta_str = f"  ({'+' if delta >= 0 else ''}{delta:.4f} vs baseline)"

        print(
            f"  {label:<30s} | {n_epochs:>6d} | {_fmt(final_loss):>10s} "
            f"| {_fmt(ap):>8s} | {_fmt(ap50):>8s} "
            f"| {_fmt(ap_s):>9s} | {_fmt(ap_l):>9s}{delta_str}"
        )

    print("=" * 90)


def analyze_loss_components(metrics, label):
    """Analyze loss component proportions and detect dominance issues.

    Note: RT-DETR's total training loss includes auxiliary decoder losses
    (aux_0 ... aux_4), denoising losses (dn_0 ... dn_5), and encoder
    losses (enc_0), which together account for the majority of the total.
    To avoid misleading ratios, component proportions are computed
    relative to the sum of the main loss terms only.

    Args:
        metrics: Metrics dict for one experiment.
        label:   Experiment label.
    """
    print(f"\n--- 损失分量分析 / Loss Component Analysis: {label} ---")

    # Look at the final epoch
    idx = -1
    total = metrics["train_loss"][idx]
    vfl = metrics["loss_vfl"][idx]
    bbox = metrics["loss_bbox"][idx]
    giou = metrics["loss_giou"][idx]
    nwd = metrics["loss_nwd"][idx]

    if math.isnan(total) or total == 0:
        print("  No valid loss data found.")
        return

    main_components = {"VFL (分类)": vfl, "BBox L1 (坐标)": bbox, "GIoU": giou}
    if not math.isnan(nwd):
        main_components["NWD"] = nwd

    # Sum of main terms only (total also includes auxiliary decoder/dn/enc losses)
    valid_main_vals = [v for v in main_components.values() if not math.isnan(v)]
    main_sum = sum(valid_main_vals) if valid_main_vals else 1.0

    print(f"  Total training loss (incl. aux/dn/enc): {total:.4f}")
    print(f"  Main loss terms sum (vfl+bbox+giou[+nwd]): {main_sum:.4f}")
    print(
        f"  Note: aux/dn/enc losses make up "
        f"{(total - main_sum) / total * 100:.1f}% of total."
    )
    print()
    print(f"  {'Component':<20s} {'Value':>10s} {'% of main':>12s} {'Status':>20s}")
    print(f"  {'-'*66}")
    for name, val in main_components.items():
        if math.isnan(val):
            continue
        ratio = val / main_sum * 100 if main_sum > 0 else 0
        if ratio < 3:
            status = "⚠ 贡献过低 (<3%)"
        elif ratio > 65:
            status = "⚠ 过度主导 (>65%)"
        else:
            status = "✓ 正常"
        print(f"  {name:<20s} {val:>10.4f} {ratio:>11.1f}% {status:>20s}")

    # Check GIoU vs NWD conflict
    if not math.isnan(nwd) and not math.isnan(giou):
        ratio = nwd / giou if giou > 0 else float("inf")
        print(f"\n  NWD / GIoU 比值: {ratio:.3f}")
        if ratio > 1.5:
            print("  ⚠ 警告: NWD 损失远大于 GIoU，可能导致梯度竞争过强")
        elif ratio < 0.3:
            print("  ⚠ 警告: NWD 损失过小，其梯度信号可能被 GIoU 压制")
        else:
            print("  ✓ NWD 与 GIoU 比值处于合理范围")


def compute_convergence_stats(metrics, label):
    """Compute convergence statistics for an experiment.

    Args:
        metrics: Metrics dict for one experiment.
        label:   Experiment label.
    """
    print(f"\n--- 收敛分析 / Convergence Analysis: {label} ---")

    ap_values = [v for v in metrics["ap"] if not math.isnan(v)]
    loss_values = [v for v in metrics["train_loss"] if not math.isnan(v)]

    if not ap_values:
        print("  No AP data available.")
        return

    peak_ap = max(ap_values)
    final_ap = ap_values[-1]
    peak_epoch_idx = ap_values.index(peak_ap)
    peak_epoch = metrics["epochs"][peak_epoch_idx] if peak_epoch_idx < len(metrics["epochs"]) else peak_epoch_idx

    print(f"  Peak AP: {peak_ap:.4f} (at epoch {peak_epoch})")
    print(f"  Final AP: {final_ap:.4f}")

    if peak_ap > final_ap + 0.005:
        print(
            f"  ⚠ 警告: 模型在 epoch {peak_epoch} 后出现性能下降 "
            f"({peak_ap:.4f} → {final_ap:.4f})，可能存在过拟合"
        )
    else:
        print("  ✓ 模型收敛稳定，未出现明显过拟合")

    # Check loss trajectory
    if len(loss_values) >= 10:
        early_loss = sum(loss_values[:5]) / 5
        late_loss = sum(loss_values[-5:]) / 5
        reduction = (early_loss - late_loss) / early_loss * 100
        print(f"  Loss 下降幅度: {early_loss:.4f} → {late_loss:.4f} ({reduction:.1f}%)")
        if reduction < 20:
            print("  ⚠ 警告: Loss 下降幅度不足 20%，模型可能未充分收敛（建议增加训练轮数）")


def detect_gradient_conflict_indicators(all_metrics, labels):
    """Detect indirect indicators of gradient conflict from loss curves.

    Specifically, checks whether the combined model's loss is higher than
    any individual model's loss in the final epochs, which would suggest
    optimization difficulties.

    Args:
        all_metrics: List of metrics dicts.
        labels:      List of experiment labels.
    """
    print("\n--- 梯度冲突间接指标 / Indirect Gradient Conflict Indicators ---")

    if len(all_metrics) < 2:
        print("  需要至少 2 个实验日志才能进行对比分析。")
        return

    final_aps = []
    final_losses = []
    for metrics in all_metrics:
        final_aps.append(_safe_last(metrics["ap"]))
        final_losses.append(_safe_last(metrics["train_loss"]))

    # Find combined (last) and compare with individual experiments
    combined_idx = len(all_metrics) - 1
    combined_ap = final_aps[combined_idx]
    individual_aps = final_aps[:combined_idx]

    if not math.isnan(combined_ap) and individual_aps:
        valid_individual = [ap for ap in individual_aps if not math.isnan(ap)]
        if valid_individual:
            best_individual = max(valid_individual)
            sum_gains = sum(
                ap - valid_individual[0] for ap in valid_individual[1:]
            ) if len(valid_individual) > 1 else 0

            print(f"  最佳单独改进 AP: {best_individual:.4f} ({labels[final_aps.index(best_individual)]})")
            print(f"  组合改进 AP ({labels[combined_idx]}): {combined_ap:.4f}")

            if combined_ap < best_individual:
                print("  ⚠ 严重警告: 组合改进效果不如最佳单独改进！")
                print("    可能原因: 梯度冲突导致优化方向错误，建议采用分阶段训练策略。")
            elif combined_ap < best_individual + 0.005:
                print("  ⚠ 警告: 组合改进相比最佳单独改进仅有微小提升。")
                print("    建议: 检查损失权重配置，尝试降低 GIoU 权重（1 → 0.5）。")
            else:
                print("  ✓ 组合改进超过了最佳单独改进。")


def print_recommendations(all_metrics, labels):
    """Print optimization recommendations based on analysis results."""
    print("\n" + "=" * 70)
    print("优化建议 / Optimization Recommendations")
    print("=" * 70)

    print("""
【建议 1】调整损失权重（立即可做）
  将配置文件中的损失权重修改为：
    loss_vfl: 1
    loss_bbox: 5
    loss_giou: 1      # 从 2 降到 1，减少与 NWD 的梯度竞争
    loss_nwd: 1.0     # NWD 权重设为 1

【建议 2】分阶段训练（最推荐）
  Stage 1 (0-30 epochs):  Baseline + LS 卷积（建立稳定特征基础）
  Stage 2 (30-60 epochs): + 动态查询分组（在稳定特征上训练查询）
  Stage 3 (60-90 epochs): + NWD 损失（最后引入损失函数改进）

【建议 3】延长训练（快速验证）
  组合改进的收敛速度慢于单独改进，建议将训练 epoch 从 72 增加到 100+

【建议 4】NWD 权重预热（中等复杂度）
  在前 20 个 epoch 将 loss_nwd 权重线性从 0 增加到目标值，
  避免训练初期 NWD 梯度干扰已收敛的 GIoU 优化方向

【建议 5】检查小目标 AP
  VisDrone 小目标 AP 是 NWD 损失最关键的指标，
  如果组合版本的 AP_small 没有提升，说明 NWD 的梯度被其他损失压制
""")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RT-DETR training logs to diagnose combined improvement conflicts"
    )
    parser.add_argument(
        "--log-files",
        nargs="+",
        required=True,
        help="Paths to JSONL training log files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each log file (default: filename without extension)",
    )
    args = parser.parse_args()

    log_files = args.log_files
    labels = args.labels
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in log_files]

    if len(labels) != len(log_files):
        print(
            f"Error: number of --labels ({len(labels)}) must match "
            f"number of --log-files ({len(log_files)})"
        )
        sys.exit(1)

    print(f"\n正在分析 {len(log_files)} 个日志文件 / Analyzing {len(log_files)} log files...\n")

    all_metrics = []
    for filepath, label in zip(log_files, labels):
        if not os.path.isfile(filepath):
            print(f"  Error: log file not found: {filepath}")
            sys.exit(1)
        records = parse_log_file(filepath)
        print(f"  Loaded {len(records)} epoch records from {filepath}")
        metrics = extract_metrics(records)
        all_metrics.append(metrics)

    # Main comparison table
    compare_experiments(all_metrics, labels)

    # Per-experiment detailed analysis (always run both analyses by default,
    # but individual flags remain available for documentation purposes)
    for label, metrics in zip(labels, all_metrics):
        analyze_loss_components(metrics, label)
        compute_convergence_stats(metrics, label)

    # Cross-experiment conflict detection
    detect_gradient_conflict_indicators(all_metrics, labels)

    # Print recommendations
    print_recommendations(all_metrics, labels)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_logs.py — Training log analysis tool for RT-DETR ablation experiments.

Reads JSONL training logs produced by tools/train.py and produces:
  - Per-epoch loss curves (train_loss, loss_vfl, loss_bbox, loss_giou, loss_nwd)
  - COCO metric curves (AP, AP50, AP75, APsmall, APmedium, APlarge)
  - Side-by-side comparison table of multiple experiments
  - Loss contribution analysis (percentage of each loss term)
  - Convergence diagnostics (gradient instability, loss spikes)

Usage:
    python tools/analyze_logs.py \
        --logs output/ablation/baseline/log.txt \
                output/ablation/nwd_only/log.txt \
                output/rtdetrv2_r50vd_visdrone_combined/log.txt \
        --names baseline nwd_only combined \
        --output_dir /tmp/analysis_output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_log(log_path: str) -> List[dict]:
    """Parse a JSONL training log file and return a list of epoch records."""
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] {log_path}:{line_no} — skipping invalid JSON: {exc}",
                    file=sys.stderr,
                )
    return records


def extract_metric(records: List[dict], key: str) -> List[Optional[float]]:
    """Extract a scalar metric for every epoch (None if key is absent)."""
    return [r.get(key) for r in records]


def extract_coco_metric(
    records: List[dict], coco_key: str = "test_coco_eval_bbox", index: int = 0
) -> List[Optional[float]]:
    """Extract a single value from the COCO metrics array."""
    result = []
    for r in records:
        arr = r.get(coco_key)
        if arr is not None and len(arr) > index:
            result.append(arr[index])
        else:
            result.append(None)
    return result


# COCO metric index mapping
COCO_METRICS = {
    "AP":       0,
    "AP50":     1,
    "AP75":     2,
    "APsmall":  3,
    "APmedium": 4,
    "APlarge":  5,
    "AR1":      6,
    "AR10":     7,
    "AR100":    8,
    "ARsmall":  9,
    "ARmedium": 10,
    "ARlarge":  11,
}


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_loss_contribution(records: List[dict]) -> Dict[str, float]:
    """
    Compute the average percentage contribution of each named loss term
    to the total train_loss over the last 10 epochs.
    """
    recent = records[-10:] if len(records) >= 10 else records
    contributions: Dict[str, List[float]] = {}

    for r in recent:
        total = r.get("train_loss")
        if total is None or total == 0:
            continue
        for key, val in r.items():
            if key.startswith("train_loss_") and val is not None:
                short = key.replace("train_loss_", "")
                contributions.setdefault(short, []).append(val / total * 100)

    return {k: sum(v) / len(v) for k, v in contributions.items() if v}


def detect_loss_spikes(
    records: List[dict],
    key: str = "train_loss",
    threshold: float = 0.2,
) -> List[int]:
    """
    Return epoch indices where the loss jumped by more than `threshold`
    relative change compared to the previous epoch (likely gradient explosion).
    """
    spikes = []
    values = extract_metric(records, key)
    for i in range(1, len(values)):
        prev, curr = values[i - 1], values[i]
        if prev is None or curr is None or prev == 0:
            continue
        rel_change = (curr - prev) / abs(prev)
        if rel_change > threshold:
            epoch = records[i].get("epoch", i)
            spikes.append(epoch)
    return spikes


def summarize_experiment(name: str, records: List[dict]) -> Dict[str, object]:
    """Build a summary dict for a single experiment."""
    if not records:
        return {"name": name, "error": "empty log"}

    epochs = [r.get("epoch", i) for i, r in enumerate(records)]
    final = records[-1]

    # Best AP (ignoring None)
    aps = extract_coco_metric(records, index=COCO_METRICS["AP"])
    valid_aps = [(e, v) for e, v in zip(epochs, aps) if v is not None]
    best_epoch, best_ap = max(valid_aps, key=lambda x: x[1]) if valid_aps else (None, None)

    final_ap = valid_aps[-1][1] if valid_aps else None
    final_loss = final.get("train_loss")

    # Loss contributions
    contributions = compute_loss_contribution(records)

    # Convergence: is loss still decreasing at the end?
    losses = [r.get("train_loss") for r in records[-5:] if r.get("train_loss") is not None]
    converged = len(losses) >= 2 and losses[-1] <= losses[0]

    # Spikes
    spikes = detect_loss_spikes(records)

    return {
        "name": name,
        "total_epochs": len(records),
        "final_epoch": epochs[-1],
        "final_train_loss": final_loss,
        "final_ap": final_ap,
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "loss_contributions_%": contributions,
        "converged": converged,
        "loss_spikes_at_epochs": spikes,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def print_comparison_table(summaries: List[Dict]) -> None:
    """Print a side-by-side comparison table to stdout."""
    headers = [
        "Experiment", "Epochs", "Final Loss", "Final AP", "Best AP",
        "Best Epoch", "Converged", "Spikes",
    ]
    col_widths = [20, 7, 12, 10, 8, 11, 10, 10]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(
        f" {h:<{w}} " for h, w in zip(headers, col_widths)
    ) + "|"

    print()
    print(_c("=" * 80, _BOLD))
    print(_c("  EXPERIMENT COMPARISON", _BOLD))
    print(_c("=" * 80, _BOLD))
    print(sep)
    print(_c(header_row, _BOLD))
    print(sep)

    best_ap_value = max(
        (s.get("final_ap") or 0) for s in summaries
    )

    for s in summaries:
        final_ap_str = f"{s['final_ap']:.4f}" if s.get("final_ap") is not None else "N/A"
        best_ap_str  = f"{s['best_ap']:.4f}"  if s.get("best_ap")  is not None else "N/A"

        # Highlight best result
        ap_color = _GREEN if s.get("final_ap") == best_ap_value else _RESET

        row = "|" + "|".join([
            f" {s['name']:<{col_widths[0]}} ",
            f" {str(s.get('total_epochs', 'N/A')):<{col_widths[1]}} ",
            (
                f" {s['final_train_loss']:<{col_widths[2]}.4f} "
                if isinstance(s.get("final_train_loss"), (int, float))
                else f" {'N/A':<{col_widths[2]}} "
            ),
            f" {_c(final_ap_str, ap_color):<{col_widths[3] + len(ap_color) + len(_RESET)}} ",
            f" {best_ap_str:<{col_widths[4]}} ",
            f" {str(s.get('best_epoch', 'N/A')):<{col_widths[5]}} ",
            f" {('Yes' if s.get('converged') else 'No'):<{col_widths[6]}} ",
            f" {str(len(s.get('loss_spikes_at_epochs', []))):<{col_widths[7]}} ",
        ]) + "|"
        print(row)

    print(sep)


def print_loss_contribution_report(summaries: List[Dict]) -> None:
    """Print loss term contribution breakdown for each experiment."""
    print()
    print(_c("=" * 80, _BOLD))
    print(_c("  LOSS CONTRIBUTION ANALYSIS (last 10 epochs avg, %)", _BOLD))
    print(_c("=" * 80, _BOLD))

    all_terms = set()
    for s in summaries:
        all_terms.update(s.get("loss_contributions_%", {}).keys())
    all_terms = sorted(all_terms)

    name_w = 22
    term_w = 12
    header = f"  {'Experiment':<{name_w}}" + "".join(
        f"  {t:<{term_w}}" for t in all_terms
    )
    print(header)
    print("  " + "-" * (name_w + (term_w + 2) * len(all_terms)))

    for s in summaries:
        contrib = s.get("loss_contributions_%", {})
        row = f"  {s['name']:<{name_w}}"
        for t in all_terms:
            val = contrib.get(t)
            if val is None:
                row += f"  {'N/A':<{term_w}}"
            elif val < 5:
                row += f"  {_c(f'{val:.1f}%', _YELLOW):<{term_w + len(_YELLOW) + len(_RESET)}}"
            elif val > 50:
                row += f"  {_c(f'{val:.1f}%', _RED):<{term_w + len(_RED) + len(_RESET)}}"
            else:
                row += f"  {f'{val:.1f}%':<{term_w}}"
        print(row)

    print()
    print("  Legend:")
    print(f"    {_c('<5%', _YELLOW)}  — loss term may be suppressed (too low weight)")
    print(f"    {_c('>50%', _RED)} — loss term may dominate training")


def print_diagnostic_warnings(summaries: List[Dict]) -> None:
    """Print actionable warnings for each experiment."""
    print()
    print(_c("=" * 80, _BOLD))
    print(_c("  DIAGNOSTIC WARNINGS", _BOLD))
    print(_c("=" * 80, _BOLD))

    any_warning = False
    for s in summaries:
        warnings = []

        spikes = s.get("loss_spikes_at_epochs", [])
        if spikes:
            warnings.append(
                f"Loss spikes detected at epochs {spikes}. "
                "Possible gradient explosion — consider reducing lr or increasing warmup."
            )

        if not s.get("converged"):
            warnings.append(
                "Loss did NOT converge in the last 5 epochs. "
                "Model may need more training epochs or a lower learning rate."
            )

        contrib = s.get("loss_contributions_%", {})
        for term, pct in contrib.items():
            if pct < 5:
                warnings.append(
                    f"Loss term '{term}' contributes only {pct:.1f}% — "
                    "its weight may be too small or the term rarely fires."
                )
            if pct > 50:
                warnings.append(
                    f"Loss term '{term}' contributes {pct:.1f}% — "
                    "it dominates training and may suppress other objectives."
                )

        if warnings:
            any_warning = True
            print(f"\n  [{s['name']}]")
            for w in warnings:
                print(f"    ⚠  {w}")

    if not any_warning:
        print("  No critical warnings detected.")


def print_improvement_recommendations(summaries: List[Dict]) -> None:
    """Print specific recommendations based on the combined analysis."""
    print()
    print(_c("=" * 80, _BOLD))
    print(_c("  IMPROVEMENT RECOMMENDATIONS", _BOLD))
    print(_c("=" * 80, _BOLD))

    names = {s["name"]: s for s in summaries}

    # Find baseline
    baseline_ap = None
    for key in ("baseline", "base"):
        if key in names and names[key].get("final_ap") is not None:
            baseline_ap = names[key]["final_ap"]
            break

    if baseline_ap is None and summaries:
        baseline_ap = min(
            (s["final_ap"] for s in summaries if s.get("final_ap") is not None),
            default=None,
        )

    print()
    print("  Based on the ablation results above, consider the following actions:")
    print()
    print("  1. LOSS WEIGHT REBALANCING")
    print("     When combining NWD + DQG, the NWD loss gradient can be amplified")
    print("     because DQG matches more queries per image. Reduce loss_nwd weight")
    print("     from 2 → 1 in the combined config:")
    print("       weight_dict: {loss_vfl: 1, loss_bbox: 5, loss_giou: 2, loss_nwd: 1}")
    print()
    print("  2. EXTENDED LEARNING RATE WARMUP")
    print("     Combined models have a more complex loss landscape. Increase")
    print("     warmup_duration from 2000 → 4000 steps to stabilize early training.")
    print()
    print("  3. ADJUSTED LR DECAY SCHEDULE")
    print("     Shift milestones from [50, 65] → [55, 68] to give the model more")
    print("     time at the initial learning rate before decaying.")
    print()
    print("  4. STAGED TRAINING STRATEGY")
    print("     Instead of enabling all three improvements from epoch 0, consider:")
    print("       Epochs  0-25: baseline + LS Conv (stable feature learning)")
    print("       Epochs 25-50: add DQG (better query assignment)")
    print("       Epochs 50-72: add NWD loss (fine-tune regression)")
    print("     Use configs/rtdetrv2/rtdetrv2_r50vd_visdrone_combined.yml which")
    print("     applies these fixes in a single run.")
    print()
    print("  5. PAIRWISE COMBINATION CHECK")
    print("     Run ablation/rtdetrv2_r50vd_visdrone_nwd_dqg.yml to isolate whether")
    print("     the NWD–DQG interaction is the primary source of degradation.")
    print()
    print("  See configs/rtdetrv2/rtdetrv2_r50vd_visdrone_combined.yml for the")
    print("  optimized combined config implementing fixes 1–3.")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(summaries: List[Dict], output_path: str) -> None:
    """Export the summary table to a CSV file."""
    import csv

    all_terms = set()
    for s in summaries:
        all_terms.update(s.get("loss_contributions_%", {}).keys())
    all_terms = sorted(all_terms)

    fieldnames = [
        "name", "total_epochs", "final_train_loss",
        "final_ap", "best_ap", "best_epoch",
        "converged", "loss_spike_count",
    ] + [f"loss_{t}_pct" for t in all_terms]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            row = {
                "name":            s["name"],
                "total_epochs":    s.get("total_epochs"),
                "final_train_loss": s.get("final_train_loss"),
                "final_ap":        s.get("final_ap"),
                "best_ap":         s.get("best_ap"),
                "best_epoch":      s.get("best_epoch"),
                "converged":       s.get("converged"),
                "loss_spike_count": len(s.get("loss_spikes_at_epochs", [])),
            }
            for t in all_terms:
                row[f"loss_{t}_pct"] = s.get("loss_contributions_%", {}).get(t)
            writer.writerow(row)

    print(f"\n  CSV summary written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and compare RT-DETR training logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more JSONL log files to analyze.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        metavar="NAME",
        help="Display names for each log (default: derived from file paths).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        metavar="DIR",
        help="Directory to write the CSV summary (optional).",
    )
    parser.add_argument(
        "--coco_key",
        default="test_coco_eval_bbox",
        help="Key in the log records containing the COCO metrics array.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve display names
    names = args.names or []
    if len(names) < len(args.logs):
        for path in args.logs[len(names):]:
            names.append(Path(path).parent.name or Path(path).stem)

    # Parse logs and build summaries
    summaries = []
    for path, name in zip(args.logs, names):
        if not os.path.isfile(path):
            print(f"[ERROR] Log file not found: {path}", file=sys.stderr)
            summaries.append({"name": name, "error": "file not found"})
            continue
        records = parse_log(path)
        summaries.append(summarize_experiment(name, records))

    # Print reports
    print_comparison_table(summaries)
    print_loss_contribution_report(summaries)
    print_diagnostic_warnings(summaries)
    print_improvement_recommendations(summaries)

    # Optional CSV export
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "ablation_summary.csv")
        export_csv(summaries, csv_path)


if __name__ == "__main__":
    main()

# Analysis: Why Combined Improvements Underperform on VisDrone

## Overview

This document analyzes why three individually effective improvements—**NWD Loss**,
**Dynamic Query Grouping (DQG)**, and **LS Convolution**—produce subadditive gains
when combined on the RT-DETR model trained on the VisDrone dataset.

---

## 1. Performance Summary

| Configuration        | AP (approx.) | Delta vs Baseline |
|----------------------|--------------|-------------------|
| Baseline             | ~0.344       | —                 |
| NWD Loss only        | ~0.356       | +0.012 (+3.5%)    |
| DQG only             | ~0.358       | +0.014 (+4.1%)    |
| LS Conv only         | ~0.350       | +0.006 (+1.7%)    |
| Expected (additive)  | ~0.376       | +0.032 (+9.3%)    |
| **Actual (combined)**| ~0.362       | **+0.018 (+5.2%)**|

The combined model recovers only ~56% of the theoretically additive gain.

---

## 2. Root Cause Analysis

### 2.1 Loss Weight Imbalance (Primary Cause)

The baseline criterion uses:

```yaml
weight_dict: {loss_vfl: 1, loss_bbox: 5, loss_giou: 2}
# Classification : Regression = 1 : 7
```

Adding NWD loss at weight 2 yields:

```yaml
weight_dict: {loss_vfl: 1, loss_bbox: 5, loss_giou: 2, loss_nwd: 2}
# Classification : Regression = 1 : 9  (+28% shift in balance)
```

When DQG is also active, the decoder assigns **more queries per image** (agnostic
mode matches queries more broadly). This amplifies the NWD gradient because NWD
is computed for every matched query-GT pair. The result is that the regression
objective inadvertently dominates, suppressing the classification signal
(`loss_vfl`) and causing AP degradation in categories with ambiguous localization.

**Diagnosis indicator:** In the combined log, `loss_vfl` plateaus earlier and at a
higher value than in the NWD-only or DQG-only runs.

### 2.2 Optimizer Instability from Simultaneous Changes (Secondary Cause)

Each improvement reshapes the loss landscape:

- **LS Conv** changes the feature distribution in `HybridEncoder` from epoch 0.
- **DQG** changes which queries receive gradients from epoch 0.
- **NWD** changes the gradient magnitude of the box head from epoch 0.

With a 2000-step warmup at `lr=0.0001`, the model simultaneously adapts to all
three changes, leading to higher gradient variance in early training. This
sometimes pushes the optimizer toward a different local minimum than any
single-improvement run reaches.

**Diagnosis indicator:** Loss spikes (>20% relative increase) at epochs 3–8 in the
combined training log that are absent in individual logs.

### 2.3 LS Conv Feature Incompatibility with NWD (Tertiary Cause)

NWD loss measures the Wasserstein distance between predicted and ground-truth box
distributions, weighted by a Gaussian kernel. Its gradients are sensitive to the
scale of predicted boxes. LS Conv changes the activation statistics of the encoder
features that feed into the box head. Without re-calibration of the NWD kernel
parameter `σ`, the loss can under- or over-penalize imprecise predictions.

**Diagnosis indicator:** `loss_nwd` converges to a different absolute scale in the
combined run vs the NWD-only run, even though `loss_giou` is similar.

### 2.4 Diminishing Returns from Overlapping Optimization Targets

All three improvements ultimately target **small object detection quality**:

- NWD: better regression loss for small, hard-to-localize objects.
- DQG: better query assignment so small objects are not missed.
- LS Conv: richer multi-scale features for small object representation.

Once one improvement covers a gap, the others have less room to improve.
For VisDrone (densely packed small objects), DQG alone already captures most
of the assignment benefit; NWD then adds less on top.

---

## 3. Proposed Fixes

### Fix 1 — Rebalance NWD Loss Weight

Reduce `loss_nwd` from 2 → 1 when using DQG to maintain the original
classification-to-regression ratio:

```yaml
# configs/rtdetrv2/rtdetrv2_r50vd_visdrone_combined.yml
RTDETRCriterionv2:
  weight_dict: {loss_vfl: 1, loss_bbox: 5, loss_giou: 2, loss_nwd: 1}
  losses: ['vfl', 'boxes', 'nwd']
```

Ratio check: vfl : box = 1 : (5+2+1) = 1:8, close to baseline 1:7. ✓

### Fix 2 — Extend Learning Rate Warmup

Double the warmup duration to give the optimizer time to stabilize before all
three objectives interact at full scale:

```yaml
lr_warmup_scheduler:
  type: LinearWarmup
  warmup_duration: 4000   # was 2000
```

### Fix 3 — Shift LR Decay Milestones

Delay the LR decay steps to give the combined model more convergence time at the
initial learning rate:

```yaml
lr_scheduler:
  type: MultiStepLR
  milestones: [55, 68]   # was [50, 65]
  gamma: 0.1
```

All three fixes are applied in `configs/rtdetrv2/rtdetrv2_r50vd_visdrone_combined.yml`.

---

## 4. Ablation Study Design

Run the following 8 configurations to isolate and confirm each interaction effect.
All configs are in `configs/rtdetrv2/ablation/`.

| Config file                              | Purpose                              |
|------------------------------------------|--------------------------------------|
| `rtdetrv2_r50vd_visdrone_baseline.yml`   | Reference point                      |
| `rtdetrv2_r50vd_visdrone_nwd.yml`        | NWD effect in isolation              |
| `rtdetrv2_r50vd_visdrone_dqg.yml`        | DQG effect in isolation              |
| `rtdetrv2_r50vd_visdrone_ls.yml`         | LS Conv effect in isolation          |
| `rtdetrv2_r50vd_visdrone_nwd_dqg.yml`    | NWD + DQG interaction                |
| `rtdetrv2_r50vd_visdrone_nwd_ls.yml`     | NWD + LS Conv interaction            |
| `rtdetrv2_r50vd_visdrone_dqg_ls.yml`     | DQG + LS Conv interaction            |
| `rtdetrv2_r50vd_visdrone_all_naive.yml`  | All three (naive, no fixes)          |

After training, run the analysis tool to compare all experiments:

```bash
python tools/analyze_logs.py \
    --logs \
        output/ablation/baseline/log.txt \
        output/ablation/nwd_only/log.txt \
        output/ablation/dqg_only/log.txt \
        output/ablation/ls_conv_only/log.txt \
        output/ablation/nwd_dqg/log.txt \
        output/ablation/nwd_ls/log.txt \
        output/ablation/dqg_ls/log.txt \
        output/ablation/all_naive/log.txt \
    --names baseline nwd dqg ls nwd+dqg nwd+ls dqg+ls all_naive \
    --output_dir /tmp/ablation_results
```

### Expected interaction matrix

| Pair         | Expected synergy | Potential conflict              |
|--------------|------------------|---------------------------------|
| NWD + DQG    | Low–Medium       | DQG amplifies NWD gradient      |
| NWD + LS     | Medium           | LS changes feature scale for NWD|
| DQG + LS     | High             | Largely orthogonal improvements |
| All three    | Medium (w/ fixes)| Use optimized combined config   |

---

## 5. Alternative Training Strategy: Staged Integration

If the optimized combined config still underperforms, try staged training where
each improvement is added incrementally:

```
Stage 1 (Epochs 0–25):   Baseline + LS Conv
  → Let the model learn stable features with the new conv structure.

Stage 2 (Epochs 25–50):  + Dynamic Query Grouping
  → Adapt query assignment on top of the new features.

Stage 3 (Epochs 50–72):  + NWD Loss (weight=1)
  → Fine-tune regression precision as the model nears convergence.
```

This avoids the simultaneous gradient conflicts of the naive approach.

---

## 6. Checklist for Future Improvement Combinations

When combining multiple improvements, always verify:

- [ ] **Loss weight ratio**: total_regression_weight / classification_weight ≤ 9
      (baseline ratio is 7; stay within ±30%).
- [ ] **Gradient scale**: monitor `loss_nwd` / `loss_giou` ratio — should stay < 1.5.
- [ ] **Early loss spikes**: check epochs 1–10 for >20% relative loss jumps.
- [ ] **Convergence**: loss should decrease monotonically in last 5 epochs.
- [ ] **Pairwise ablation**: confirm each pair of improvements before combining all.
- [ ] **Warmup duration**: increase warmup by 1000 steps for each additional
      improvement module beyond the first.

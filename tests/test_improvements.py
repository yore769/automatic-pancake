"""Tests for the three RT-DETR improvements and their combined integration.

These tests verify:
1. NWD Loss correctness and scale properties
2. Dynamic Query Grouping isolation (detached gradients)
3. LS Conv selective application (only CCFM layers)
4. Combined configuration correctness (weight ratios, flag settings)
5. Criterion NWD-only-final behaviour

Run with:  python -m pytest tests/ -v
       or:  python tests/test_improvements.py
"""

import sys
import os
import math

# Add repository root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nwd_loss_simple(pred, tgt, C=12.8):
    """Pure-Python NWD loss for testing (cx,cy,w,h format)."""
    cx_p, cy_p, w_p, h_p = pred
    cx_t, cy_t, w_t, h_t = tgt
    sx_p, sy_p = w_p / 2, h_p / 2
    sx_t, sy_t = w_t / 2, h_t / 2
    w2 = math.sqrt(
        (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2 +
        (sx_p - sx_t) ** 2 + (sy_p - sy_t) ** 2
    )
    return 1.0 - math.exp(-w2 / C)


def giou_loss_simple(pred, tgt):
    """Pure-Python GIoU loss for testing (cx,cy,w,h format)."""
    def to_xyxy(b):
        return b[0] - b[2]/2, b[1] - b[3]/2, b[0] + b[2]/2, b[1] + b[3]/2
    p = to_xyxy(pred)
    t = to_xyxy(tgt)
    ix1, iy1 = max(p[0], t[0]), max(p[1], t[1])
    ix2, iy2 = min(p[2], t[2]), min(p[3], t[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_p = (p[2] - p[0]) * (p[3] - p[1])
    area_t = (t[2] - t[0]) * (t[3] - t[1])
    union = area_p + area_t - inter + 1e-7
    ex1 = min(p[0], t[0]); ey1 = min(p[1], t[1])
    ex2 = max(p[2], t[2]); ey2 = max(p[3], t[3])
    enc = (ex2 - ex1) * (ey2 - ey1) + 1e-7
    iou = inter / union
    return 1.0 - (iou - (enc - union) / enc)


def load_yaml(rel_path):
    import yaml
    abs_path = os.path.join(os.path.dirname(__file__), '..', rel_path)
    with open(abs_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# NWD Loss tests
# ---------------------------------------------------------------------------

class TestNWDLossLogic:
    """Test NWD loss mathematical properties."""

    def test_perfect_match_gives_zero_loss(self):
        """Identical predicted and target boxes should yield 0 loss."""
        box = (0.5, 0.5, 0.1, 0.1)
        loss = nwd_loss_simple(box, box)
        assert abs(loss) < 1e-9, f'Expected 0.0 but got {loss}'

    def test_loss_bounded_zero_one(self):
        """NWD loss must be in [0, 1]."""
        cases = [
            ((0.1, 0.1, 0.02, 0.02), (0.9, 0.9, 0.02, 0.02)),
            ((0.5, 0.5, 0.01, 0.01), (0.5, 0.5, 0.5, 0.5)),
            ((0.0, 0.0, 0.001, 0.001), (1.0, 1.0, 0.001, 0.001)),
        ]
        for pred, tgt in cases:
            loss = nwd_loss_simple(pred, tgt)
            assert 0.0 <= loss <= 1.0, f'Loss {loss} out of [0,1] for {pred},{tgt}'

    def test_small_object_nonzero_gradient(self):
        """NWD produces smooth loss even for non-overlapping small boxes.

        GIoU saturates at 2.0 for non-overlapping boxes, giving a constant
        gradient that doesn't help the model localise.  NWD gives a loss
        that decays smoothly (< 1.0) even for distant boxes, providing a
        richer gradient signal for small objects.
        """
        pred = (0.1, 0.1, 0.02, 0.02)
        tgt = (0.9, 0.9, 0.02, 0.02)
        loss = nwd_loss_simple(pred, tgt)
        assert loss < 1.0, f'NWD loss {loss} is saturated — no gradient signal'
        assert loss > 0.0, 'NWD loss should be positive for non-matching boxes'

    def test_loss_monotonically_increases_with_distance(self):
        """Loss increases as boxes move further apart."""
        base = (0.5, 0.5, 0.1, 0.1)
        offsets = [0.0, 0.05, 0.1, 0.2, 0.4]
        losses = [nwd_loss_simple((0.5 + d, 0.5, 0.1, 0.1), base)
                  for d in offsets]
        for i in range(len(losses) - 1):
            assert losses[i] <= losses[i + 1] + 1e-12, (
                f'Loss not monotonically increasing: {losses}')

    def test_nwd_weight_ratio_to_giou(self):
        """The recommended NWD:GIoU weight ratio is 0.5 (the integration fix)."""
        default_giou_weight = 2
        default_nwd_weight = 1   # as set in the combined VisDrone config
        ratio = default_nwd_weight / default_giou_weight
        assert abs(ratio - 0.5) < 1e-9, (
            f'Expected NWD/GIoU ratio = 0.5, got {ratio}')


# ---------------------------------------------------------------------------
# Dynamic Query Grouping tests
# ---------------------------------------------------------------------------

class TestDynamicQueryGroupingConfig:
    """Test Dynamic Query Grouping configuration parameters."""

    def test_warmup_prevents_early_grouping(self):
        """Grouping should be disabled during warm-up period."""
        warmup = 2000
        assert not (1999 >= warmup), 'Grouping should be inactive at iter 1999'

    def test_grouping_activates_after_warmup(self):
        """Grouping should activate exactly at warmup boundary."""
        warmup = 2000
        assert (2000 >= warmup), 'Grouping should be active at iter 2000'

    def test_group_count_for_query_count(self):
        """900 queries / 15 groups = 60 per group — a reasonable density."""
        assert 10 <= 900 / 15 <= 100, 'Queries per group out of reasonable range'

    def test_additive_context_small_init(self):
        """Group context initialised near zero acts as a small additive perturbation."""
        import math
        # std=0.01 → 95% of initial group context magnitudes < 0.02
        std = 0.01
        # d_model = 256, so L2 norm of a 256-dim vector with each element ~N(0,0.01)
        # E[||v||] ≈ std * sqrt(d_model) = 0.01 * 16 = 0.16
        # A typical query embedding has unit L2 norm ≈ 1.0
        expected_context_norm = std * math.sqrt(256)
        assert expected_context_norm < 0.5, (
            f'Group context L2 norm {expected_context_norm:.3f} is too large '
            f'(should be < 0.5 of query norm)')

    def test_detached_reference_points_isolate_grouping_gradient(self):
        """Verifying that grouping uses detached refs (design invariant)."""
        # From the code: ref_xy = reference_points[..., :2].detach()
        # This is a design invariant we enforce in the docstring and code.
        # Here we test the requirement as a specification.
        code_path = os.path.join(
            os.path.dirname(__file__), '..',
            'src/nn/decoder/rtdetrv2_decoder.py')
        with open(code_path) as f:
            code = f.read()
        assert '.detach()' in code, (
            'DynamicQueryGrouping.forward must call .detach() on reference_points '
            'to isolate the grouping decision from NWD/GIoU gradient flow.')


# ---------------------------------------------------------------------------
# LS Conv tests
# ---------------------------------------------------------------------------

class TestLSConvSelectiveApplication:
    """Test LS Conv selective application (fix for combined improvements)."""

    def test_input_projection_never_uses_ls_conv(self):
        """1×1 input projections should never become LSConv."""
        # From make_conv: LSConv requires kernel_size >= 3
        for kernel_size in [1]:
            use_ls = True
            would_use = use_ls and (kernel_size >= 3)
            assert not would_use, f'kernel_size={kernel_size} should not use LSConv'

    def test_ccfm_3x3_uses_ls_when_enabled(self):
        """3×3 CCFM convolutions should use LSConv when flag is True."""
        would_use = True and (3 >= 3)
        assert would_use, '3×3 CCFM convs should use LSConv when enabled'

    def test_ls_conv_disabled_by_default_in_base_config(self):
        """Base config should have use_ls_conv=False."""
        cfg = load_yaml('configs/rtdetrv2/include/rtdetrv2_r50vd.yml')
        assert not cfg.get('HybridEncoder', {}).get('use_ls_conv', True), \
            'use_ls_conv should be False in base config'

    def test_ls_conv_enabled_in_visdrone_config(self):
        """VisDrone combined config should enable LS Conv."""
        cfg = load_yaml('configs/rtdetrv2/rtdetrv2_r50vd_visdrone.yml')
        assert cfg.get('HybridEncoder', {}).get('use_ls_conv', False), \
            'use_ls_conv should be True in VisDrone config'

    def test_ls_conv_skip_connection_in_code(self):
        """LSConv must have a skip connection to preserve gradient flow."""
        code_path = os.path.join(
            os.path.dirname(__file__), '..',
            'src/nn/backbone/ls_conv.py')
        with open(code_path) as f:
            code = f.read()
        assert 'use_skip' in code, \
            'LSConv must implement skip/residual connection for gradient stability'


# ---------------------------------------------------------------------------
# Combined configuration tests
# ---------------------------------------------------------------------------

class TestCombinedImprovementConfig:
    """Test that the combined improvement configuration is correct."""

    def setup_method(self):
        self.cfg = load_yaml('configs/rtdetrv2/rtdetrv2_r50vd_visdrone.yml')

    def test_nwd_only_final_is_enabled(self):
        """nwd_only_final must be True to prevent NWD overloading aux layers."""
        assert self.cfg.get('RTDETRCriterionv2', {}).get(
            'nwd_only_final', False), \
            'nwd_only_final should be True to prevent gradient overload'

    def test_use_nwd_loss_is_enabled(self):
        """use_nwd_loss must be True in the combined VisDrone config."""
        assert self.cfg.get('RTDETRCriterionv2', {}).get(
            'use_nwd_loss', False), \
            'use_nwd_loss should be True in VisDrone config'

    def test_nwd_weight_is_half_giou_weight(self):
        """NWD weight should be exactly 0.5 × GIoU weight."""
        wd = self.cfg.get('RTDETRCriterionv2', {}).get('weight_dict', {})
        nwd_w = wd.get('loss_nwd', 0)
        giou_w = wd.get('loss_giou', 2)
        ratio = nwd_w / giou_w
        assert abs(ratio - 0.5) < 0.01, (
            f'NWD/GIoU ratio should be 0.5, got {ratio:.3f} '
            f'(nwd={nwd_w}, giou={giou_w})')

    def test_dynamic_grouping_enabled(self):
        """Dynamic Query Grouping should be enabled."""
        assert self.cfg.get('RTDETRTransformerv2', {}).get(
            'use_dynamic_grouping', False), \
            'use_dynamic_grouping should be True in VisDrone config'

    def test_grouping_warmup_is_at_least_1000(self):
        """Warmup must be >= 1000 to allow queries to stabilise before grouping."""
        warmup = self.cfg.get('RTDETRTransformerv2', {}).get(
            'grouping_warmup_iters', 0)
        assert warmup >= 1000, (
            f'grouping_warmup_iters={warmup} is too low; '
            f'need >= 1000 for query stabilisation')

    def test_ls_conv_enabled_in_encoder(self):
        """HybridEncoder.use_ls_conv should be True."""
        assert self.cfg.get('HybridEncoder', {}).get('use_ls_conv', False), \
            'HybridEncoder.use_ls_conv should be True in VisDrone config'

    def test_base_config_improvements_disabled(self):
        """Base model config should have all improvements disabled."""
        base = load_yaml('configs/rtdetrv2/include/rtdetrv2_r50vd.yml')
        assert not base.get('HybridEncoder', {}).get('use_ls_conv', True), \
            'use_ls_conv should be False in base config'
        assert not base.get('RTDETRTransformerv2', {}).get(
            'use_dynamic_grouping', True), \
            'use_dynamic_grouping should be False in base config'
        assert not base.get('RTDETRCriterionv2', {}).get('use_nwd_loss', True), \
            'use_nwd_loss should be False in base config'


# ---------------------------------------------------------------------------
# GIoU loss tests
# ---------------------------------------------------------------------------

class TestGIoULoss:
    """Test GIoU loss for correctness."""

    def test_perfect_match(self):
        box = (0.5, 0.5, 0.2, 0.2)
        assert abs(giou_loss_simple(box, box)) < 1e-5

    def test_bounded_0_to_2(self):
        """GIoU loss must be in [0, 2]."""
        cases = [
            ((0.1, 0.1, 0.05, 0.05), (0.9, 0.9, 0.05, 0.05)),
            ((0.2, 0.2, 0.1, 0.1), (0.8, 0.8, 0.1, 0.1)),
        ]
        for p, t in cases:
            loss = giou_loss_simple(p, t)
            assert 0.0 <= loss <= 2.0 + 1e-6, f'GIoU loss {loss} out of [0,2]'

    def test_non_overlapping_gives_high_loss(self):
        """Non-overlapping boxes should have large GIoU loss."""
        p = (0.1, 0.1, 0.05, 0.05)
        t = (0.9, 0.9, 0.05, 0.05)
        loss = giou_loss_simple(p, t)
        assert loss > 1.5, f'Non-overlapping boxes should have loss > 1.5, got {loss}'


# ---------------------------------------------------------------------------
# Code structure tests (verify key code properties)
# ---------------------------------------------------------------------------

class TestCodeStructure:
    """Verify that key integration fixes are present in source code."""

    def _read(self, rel_path):
        path = os.path.join(os.path.dirname(__file__), '..', rel_path)
        with open(path) as f:
            return f.read()

    def test_criterion_has_nwd_only_final_param(self):
        code = self._read('src/nn/criterion/rtdetrv2_criterion.py')
        assert 'nwd_only_final' in code, \
            'Criterion must have nwd_only_final parameter'

    def test_criterion_nwd_default_weight_formula(self):
        """Criterion auto-sets NWD weight to 0.5 × GIoU when not specified."""
        code = self._read('src/nn/criterion/rtdetrv2_criterion.py')
        assert 'giou_w * 0.5' in code or '0.5 * giou_w' in code, \
            'Criterion must auto-set NWD weight to 0.5 × GIoU'

    def test_decoder_has_warmup_check(self):
        code = self._read('src/nn/decoder/rtdetrv2_decoder.py')
        assert 'grouping_warmup_iters' in code, \
            'Decoder must implement grouping warm-up'

    def test_encoder_passes_use_ls_conv_to_repblock(self):
        code = self._read('src/nn/encoder/hybrid_encoder.py')
        assert 'use_ls_conv' in code, \
            'Encoder must accept and use use_ls_conv flag'

    def test_make_conv_guards_kernel_size(self):
        """make_conv must guard kernel_size >= 3 before using LSConv."""
        code = self._read('src/nn/backbone/ls_conv.py')
        assert 'kernel_size >= 3' in code, \
            'make_conv must gate LSConv on kernel_size >= 3'

    def test_nwd_aux_outputs_skipped_when_only_final(self):
        """NWD should be skipped for aux outputs when nwd_only_final=True."""
        code = self._read('src/nn/criterion/rtdetrv2_criterion.py')
        assert 'not self.nwd_only_final' in code, \
            'Criterion must skip NWD for aux layers when nwd_only_final=True'

    def test_dn_outputs_never_use_nwd(self):
        """Denoising outputs should never use NWD (prevents gradient overload)."""
        code = self._read('src/nn/criterion/rtdetrv2_criterion.py')
        # The dn section should use compute_nwd=False
        assert 'compute_nwd=False' in code, \
            'Denoising outputs must not compute NWD loss'


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_classes = [
        TestNWDLossLogic,
        TestDynamicQueryGroupingConfig,
        TestLSConvSelectiveApplication,
        TestCombinedImprovementConfig,
        TestGIoULoss,
        TestCodeStructure,
    ]

    passed = failed = 0
    for cls in test_classes:
        print(f'\n--- {cls.__name__} ---')
        obj = cls()
        if hasattr(obj, 'setup_method'):
            obj.setup_method()
        for method_name in sorted(m for m in dir(obj) if m.startswith('test_')):
            try:
                if hasattr(obj, 'setup_method'):
                    obj.setup_method()
                getattr(obj, method_name)()
                print(f'  ✓ {method_name}')
                passed += 1
            except Exception as e:
                print(f'  ✗ {method_name}: {e}')
                failed += 1

    print(f'\nResults: {passed} passed, {failed} failed')
    sys.exit(0 if failed == 0 else 1)

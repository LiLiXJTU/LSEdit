import unittest
from types import SimpleNamespace

import torch

from havedit.flux.attention_processor_headwise_subjectbg_subjectrelease import (
    HeadwiseSubjectBackgroundSubjectReleaseFluxAttnProcessor,
)
from havedit.flux.head_scores import compute_head_deviation, compute_preserve_scores


def _noop_accumulator(_value):
    return None


def _state(*, semantic_prior=None, running_semantic_prior=None):
    return SimpleNamespace(
        current_step=1,
        total_steps=4,
        text_n=1,
        latent_n=4,
        ref_n=4,
        height=2,
        width=2,
        semantic_prior=semantic_prior,
        running_semantic_prior=running_semantic_prior,
        visualize_attention=False,
        ema_delta=None,
        background_discovery_step=16,
        config=SimpleNamespace(
            runtime=SimpleNamespace(havedit_end=0),
            wsp=SimpleNamespace(warmup_steps=0),
            bhc=SimpleNamespace(enabled=True, tau_low=0.2, tau_high=0.8, lambda_max=0.5),
            trajectory_trust=SimpleNamespace(
                enabled=False,
                min_steps=0,
                release_bias=0.0,
                release_scale=1.0,
                eps=1e-6,
                ema_decay=0.9,
            ),
            havsr=SimpleNamespace(
                block_scope="double_stream",
                alpha=1.0,
                beta=2.0,
                local_kernel_size=1,
                eps=1e-6,
                threshold=0.5,
            ),
        ),
        accumulate_preserve_weight_map=_noop_accumulator,
        accumulate_edit_mask=_noop_accumulator,
    )


def _value_tensor():
    return torch.linspace(-1.0, 1.0, steps=1 * 9 * 2 * 3).reshape(1, 9, 2, 3)


class SubjectReleaseBhcScoreTests(unittest.TestCase):
    def test_no_semantic_prior_skips_bhc_instead_of_using_fallback_scores(self):
        processor = HeadwiseSubjectBackgroundSubjectReleaseFluxAttnProcessor(
            state=_state(),
            block_name="ds_0",
            is_single_stream=False,
        )

        def fail_if_called(_values, _refs, _scores):
            raise AssertionError("BHC should be skipped without subject-release scores")

        processor._apply_bhc = fail_if_called

        result = processor._maybe_replace_double_stream_values(_value_tensor())

        self.assertEqual(result.shape, _value_tensor().shape)

    def test_bhc_uses_subject_release_preserve_score_from_running_prior(self):
        running_prior = torch.tensor([[[1.0, 0.5], [0.0, 0.25]]])
        processor = HeadwiseSubjectBackgroundSubjectReleaseFluxAttnProcessor(
            state=_state(
                semantic_prior=torch.zeros_like(running_prior),
                running_semantic_prior=running_prior,
            ),
            block_name="ds_0",
            is_single_stream=False,
        )
        captured = {}

        def capture_scores(values, _refs, scores):
            captured["scores"] = scores.detach().clone()
            return values

        processor._apply_bhc = capture_scores
        value = _value_tensor()
        value_by_head = value.transpose(1, 2)
        v_pred, v_ref = processor._split_pred_and_ref_values(value_by_head)
        expected_scores = compute_preserve_scores(
            deviation=compute_head_deviation(v_pred, v_ref),
            semantic_prior=running_prior,
            height=processor.state.height,
            width=processor.state.width,
            alpha=processor.state.config.havsr.alpha,
            beta=processor.state.config.havsr.beta,
            kernel_size=processor.state.config.havsr.local_kernel_size,
            eps=processor.state.config.havsr.eps,
        )

        processor._maybe_replace_double_stream_values(value)

        self.assertIn("scores", captured)
        self.assertTrue(torch.allclose(captured["scores"], expected_scores))


if __name__ == "__main__":
    unittest.main()

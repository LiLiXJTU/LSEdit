import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from havedit.flux.attention_processor_headwise_subjectbg_subjectrelease import HAVEditFluxAttnProcessor


class _FakeAccumulator:
    def update(self, attn_probs, text_n, latent_n):
        self.last_shape = tuple(attn_probs.shape)
        self.last_text_n = text_n
        self.last_latent_n = latent_n


class _FakeState:
    def __init__(self):
        self.current_step = 0
        self.text_n = 2
        self.latent_n = 3
        self.ref_n = 3
        self.height = 1
        self.width = 3
        self.visualize_attention = False
        self.semantic_accumulator = _FakeAccumulator()
        self.latest_attention_js_ref_vs_pred = None
        self.config = SimpleNamespace(
            wsp=SimpleNamespace(warmup_steps=6),
            bhc=SimpleNamespace(enabled=False),
            havsr=SimpleNamespace(block_scope="double_stream"),
            runtime=SimpleNamespace(havedit_end=0),
        )

    def update_semantic_prior_current(self, *, attn_probs, text_n, latent_n):
        self.latest_semantic_prior_shape = tuple(attn_probs.shape)
        self.latest_semantic_prior_text_n = text_n
        self.latest_semantic_prior_latent_n = latent_n


class AttentionAccumulationTests(unittest.TestCase):
    def test_attention_accumulation_only_materializes_text_query_rows(self):
        state = _FakeState()
        processor = HAVEditFluxAttnProcessor(state=state, block_name="ds_0", is_single_stream=False)
        total_tokens = state.text_n + state.latent_n + state.ref_n
        query = torch.randn(1, total_tokens, 2, 4)
        key = torch.randn(1, total_tokens, 2, 4)

        def fake_matmul(lhs, rhs):
            self.assertEqual(lhs.shape[2], state.text_n)
            return torch.zeros(lhs.shape[0], lhs.shape[1], lhs.shape[2], rhs.shape[-1], dtype=lhs.dtype)

        with mock.patch("torch.matmul", side_effect=fake_matmul):
            processor._maybe_accumulate_from_attention(query, key, attention_mask=None)

        self.assertEqual(state.semantic_accumulator.last_shape, (1, 2, state.text_n, total_tokens))
        self.assertEqual(state.latest_semantic_prior_shape, (1, 2, state.text_n, total_tokens))


if __name__ == "__main__":
    unittest.main()

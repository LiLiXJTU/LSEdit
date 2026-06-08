import unittest

from havedit import HAVEditConfig
from havedit.flux.attention_processor_headwise_subjectbg_subjectrelease import (
    enable_headwise_subjectbg_subjectrelease_havedit,
)
from havedit.flux.warp import enable_havedit


class FakeAttention:
    def __init__(self):
        self.processor = object()
        self.original_processor = self.processor


class FakeBlock:
    def __init__(self):
        self.attn = FakeAttention()


class FakeTransformer:
    def __init__(self):
        self.transformer_blocks = [FakeBlock()]
        self.single_transformer_blocks = [FakeBlock()]


class FakePipeline:
    def __init__(self):
        self.transformer = FakeTransformer()


class WarpIntegrationTests(unittest.TestCase):
    def test_enable_havedit_replaces_processors_on_all_registered_blocks(self):
        pipeline = FakePipeline()
        config = HAVEditConfig()
        config.runtime.backend = "flux2"

        result = enable_havedit(pipeline, config)

        processors = [block.attn.processor for block in result.transformer.transformer_blocks]
        processors += [block.attn.processor for block in result.transformer.single_transformer_blocks]
        self.assertEqual(len(processors), 2)
        self.assertTrue(all(hasattr(proc, "state") for proc in processors))
        self.assertEqual(result.transformer._havedit_backend, "flux2")

    def test_subjectbg_subjectrelease_enable_sets_backend_and_stream_flags(self):
        pipeline = FakePipeline()
        config = HAVEditConfig()
        config.runtime.backend = "flux2"

        result = enable_headwise_subjectbg_subjectrelease_havedit(pipeline, config)

        self.assertEqual(result.transformer._havedit_backend, "flux2")
        ds_processor = result.transformer.transformer_blocks[0].attn.processor
        ss_processor = result.transformer.single_transformer_blocks[0].attn.processor
        self.assertIs(ds_processor.is_single_stream, False)
        self.assertIs(ss_processor.is_single_stream, True)

    def test_flux1_kontext_keeps_single_stream_processors_original(self):
        pipeline = FakePipeline()
        config = HAVEditConfig()
        config.runtime.backend = "flux1-kontext"

        result = enable_headwise_subjectbg_subjectrelease_havedit(pipeline, config)

        self.assertEqual(result.transformer._havedit_backend, "flux1-kontext")
        ds_processor = result.transformer.transformer_blocks[0].attn.processor
        ss_attention = result.transformer.single_transformer_blocks[0].attn
        self.assertTrue(hasattr(ds_processor, "state"))
        self.assertIs(ss_attention.processor, ss_attention.original_processor)


if __name__ == "__main__":
    unittest.main()

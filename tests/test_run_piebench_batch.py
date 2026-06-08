import importlib.util
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


SCRIPT_PATH = Path("/workspace/code/HAVEdit-main_changed/scripts/run_piebench_batch.py")


def load_module():
    spec = importlib.util.spec_from_file_location("run_piebench_batch", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunPiebenchBatchTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.mapping = {
            "000000000000": {
                "image_path": "0_random_140/000000000000.jpg",
                "editing_instruction": "Make the frame of the bike rusty",
                "editing_type_id": "0",
                "original_prompt": "a slanted mountain bicycle on the road in front of a building",
                "editing_prompt": "a slanted [rusty] mountain bicycle on the road in front of a building",
                "mask": [0, 10],
            },
            "000000000005": {
                "image_path": "0_random_140/000000000005.jpg",
                "editing_instruction": "Change the color of the cat from orange to black",
                "editing_type_id": "0",
                "original_prompt": "an orange cat",
                "editing_prompt": "a [black] cat",
            },
            "924000000002": {
                "image_path": "9_change_style_80/2_natural/4_outdoor/924000000002.jpg",
                "editing_instruction": "Add cartoon effect",
                "editing_type_id": "9",
            },
        }

    def test_parse_sample_ids_splits_and_trims(self):
        parsed = self.module.parse_sample_ids(" 000000000005,924000000002 ,, ")
        self.assertEqual(parsed, ["000000000005", "924000000002"])

    def test_select_entries_respects_start_and_limit(self):
        selected = self.module.select_entries(
            self.mapping,
            sample_ids=None,
            start=1,
            limit=1,
        )
        self.assertEqual([sample.sample_id for sample in selected], ["000000000005"])

    def test_select_entries_filters_by_sample_ids_in_given_order(self):
        selected = self.module.select_entries(
            self.mapping,
            sample_ids=["924000000002", "000000000000"],
            start=0,
            limit=None,
        )
        self.assertEqual(
            [sample.sample_id for sample in selected],
            ["924000000002", "000000000000"],
        )

    def test_strength_to_filename_matches_veloedit(self):
        self.assertEqual(self.module.strength_to_filename(1.0), "strength_1_00.jpg")

    def test_build_config_dir_name_uses_veloedit_style_layout(self):
        self.assertEqual(
            self.module.build_config_dir_name(
                backend="flux2",
                num_inference_steps=28,
                guidance_scale=4.0,
                seed=None,
            ),
            "havedit_flux2_klein_base_t28_g4_0_seednone",
        )

    def test_build_config_dir_name_switches_prefix_for_flux1_kontext(self):
        self.assertEqual(
            self.module.build_config_dir_name(
                backend="flux1-kontext",
                num_inference_steps=28,
                guidance_scale=4.0,
                seed=42,
            ),
            "havedit_flux1_kontext_t28_g4_0_seed42",
        )

    def test_build_config_dir_name_switches_prefix_for_qwen_image_edit(self):
        self.assertEqual(
            self.module.build_config_dir_name(
                backend="qwen-image-edit",
                num_inference_steps=28,
                guidance_scale=4.0,
                seed=42,
            ),
            "havedit_qwen_image_edit_t28_g4_0_seed42",
        )

    def test_build_sample_output_dir_matches_veloedit_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            sample = self.module._sample_from_mapping_entry(
                Path(self.module.DEFAULT_PIE_ROOT),
                "924000000002",
                self.mapping["924000000002"],
            )
            output_path = self.module.build_sample_output_dir(
                output_dir,
                "havedit_flux2_klein_base_t28_g4_0_seednone",
                sample,
            )
            self.assertEqual(
                output_path,
                output_dir
                / "havedit_flux2_klein_base_t28_g4_0_seednone"
                / "9_change_style_80/2_natural/4_outdoor"
                / "924000000002",
            )

    def test_build_sample_metadata_includes_veloedit_compatible_fields(self):
        sample = self.module._sample_from_mapping_entry(
            Path(self.module.DEFAULT_PIE_ROOT),
            "000000000000",
            self.mapping["000000000000"],
        )
        args = SimpleNamespace(
            backend="flux1-kontext",
            model_path="/workspace/model/FLUX.2-klein-base-9B",
            num_inference_steps=28,
            guidance_scale=4.0,
            seed=None,
            enable_cpu_offload=False,
            enable_bhc=True,
            enable_havedit=True,
            enable_trajectory_trust=True,
            warmup_steps=6,
            alpha=2.0,
            beta=1.0,
            threshold=0.9,
            subject_threshold=0.4,
            subject_select_mode="largest",
            subject_open_kernel=1,
            subject_close_kernel=1,
            subject_dilate_radius=1,
            background_discovery_step=16,
            subject_release_step=16,
            subject_release_scale=0.5,
        )

        metadata = self.module.build_sample_metadata(
            sample,
            args,
            generated_filename="strength_1_00.jpg",
        )

        self.assertEqual(metadata["image_id"], "000000000000")
        self.assertEqual(metadata["backend"], "flux1-kontext")
        self.assertEqual(metadata["original_path"], "0_random_140/000000000000.jpg")
        self.assertEqual(metadata["editing_instruction"], "Make the frame of the bike rusty")
        self.assertEqual(metadata["editing_type_id"], "0")
        self.assertEqual(metadata["mode"], "havedit_flux1_kontext")
        self.assertEqual(metadata["strengths"]["1.00"]["mode"], "havedit_flux1_kontext")
        self.assertEqual(metadata["strengths"]["1.00"]["path"], "strength_1_00.jpg")
        self.assertFalse(metadata["enable_cpu_offload"])
        self.assertEqual(metadata["havedit"]["threshold"], 0.9)

    def test_parser_defaults_match_demo_defaults_but_disable_cpu_offload(self):
        parser = self.module.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.backend, "flux2")
        self.assertEqual(args.model_path, "/workspace/model/FLUX.2-klein-base-9B")
        self.assertFalse(args.enable_cpu_offload)
        self.assertIsNone(args.seed)
        self.assertEqual(args.num_inference_steps, 28)
        self.assertEqual(args.guidance_scale, 4.0)
        self.assertEqual(args.warmup_steps, 6)
        self.assertEqual(args.alpha, 2.0)
        self.assertEqual(args.beta, 1.0)
        self.assertEqual(args.threshold, 0.9)

    def test_parser_uses_workspace_output_dir_by_default(self):
        parser = self.module.build_parser()
        args = parser.parse_args([])

        self.assertEqual(
            args.output_dir,
            "/workspace/code/HAVEdit-main_changed/output/piebench",
        )


if __name__ == "__main__":
    unittest.main()

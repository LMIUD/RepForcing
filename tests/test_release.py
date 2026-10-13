"""CPU-only checks for the public configuration and data interfaces."""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import lmdb
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import available, get_config


class ReleaseTests(unittest.TestCase):
    def runtime(self):
        path = ROOT / "utils/runtime.py"
        self.assertTrue(path.is_file(), "Missing shared public configuration interface")
        spec = importlib.util.spec_from_file_location("release_runtime", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def arguments(self, **changes):
        values = dict(config="dmd.framewise", config_path="", data_path=str(ROOT / "prompts/demos.txt"),
                      generator_ckpt=None, no_save=False, no_visualize=False, tf=False,
                      logdir="logs/test", wandb_save_dir="", enable_wandb=False,
                      disable_wandb=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_only_four_formal_configs_are_registered(self):
        self.assertEqual(available(), ["dmd.chunkwise", "dmd.framewise", "rft.chunkwise", "rft.framewise"])

    def test_old_stage1_names_are_not_registered(self):
        for name in ("gft.framewise", "gft.chunkwise"):
            with self.subTest(name=name), self.assertRaises(KeyError):
                get_config(name)

    def test_rft_model_registration_matches_definition(self):
        model_tree = ast.parse((ROOT / "model/diffusion_rft.py").read_text())
        model_class = next(node for node in model_tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "CausalDiffusionRFT")
        self.assertEqual([base.id for base in model_class.bases], ["BaseModel"])
        trainer_tree = ast.parse((ROOT / "trainer/rft.py").read_text())
        model_import = next(node for node in trainer_tree.body
                            if isinstance(node, ast.ImportFrom) and node.module == "model.diffusion_rft")
        self.assertEqual([alias.name for alias in model_import.names], ["CausalDiffusionRFT"])
        registry = next(node.value for node in trainer_tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "RFT_VARIANTS"
                                for target in node.targets))
        self.assertEqual([ast.literal_eval(key) for key in registry.keys], ["rft"])
        self.assertEqual([value.id for value in registry.values], ["CausalDiffusionRFT"])

    def test_formal_method_parameters(self):
        for mode, frames, timestep in (("framewise", 1, 0), ("chunkwise", 3, 0)):
            with self.subTest(mode=mode):
                self.assertIn("rft." + mode, available())
                stage1 = get_config("rft." + mode)
                stage2 = get_config("dmd." + mode)
                self.assertEqual(stage1.trainer, "rft")
                self.assertEqual((stage1.num_frame_per_block, stage1.perceptual_timestep), (frames, timestep))
                self.assertEqual(stage1.real_name, "Wan2.1-T2V-14B")
                self.assertEqual(stage1.perceptual_layer, 8)
                self.assertEqual(stage2.num_frame_per_block, frames)
                self.assertEqual(stage2.denoising_step_list, [1000, 750, 500, 250])
                self.assertFalse(stage2.generator_ckpt)

    def test_public_config_removes_nested_secrets_without_mutating_input(self):
        runtime = self.runtime()
        config = OmegaConf.create({"lr": 0.1, "wandb_key": "not-a-real-secret",
                                   "nested": {"api_key": "test-only", "enabled": True},
                                   "items": [{"access_token": "test-only", "value": 2}]})
        before = OmegaConf.to_container(config)
        self.assertEqual(runtime.public_config(config),
                         {"lr": 0.1, "nested": {"enabled": True}, "items": [{"value": 2}]})
        self.assertEqual(OmegaConf.to_container(config), before)

    def test_dmd_requires_initialization_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "generator_ckpt"):
            self.runtime().load_training_config(self.arguments())

    def test_cli_overrides_config_and_logging_defaults_off(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            ckpt = Path(directory) / "model.pt"
            ckpt.write_bytes(b"test fixture; never loaded")
            config = runtime.load_training_config(self.arguments(generator_ckpt=str(ckpt)))
            self.assertEqual(config.generator_ckpt, str(ckpt))
            self.assertEqual(config.data_path, str(ROOT / "prompts/demos.txt"))
            self.assertTrue(config.disable_wandb)

    def test_stage1_does_not_require_initialization_checkpoint(self):
        config = self.runtime().load_training_config(self.arguments(config="rft.framewise"))
        self.assertTrue(config.disable_wandb)
        self.assertEqual(config.data_path, str(ROOT / "prompts/demos.txt"))

    def test_default_yaml_location_is_independent_of_cwd(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            OmegaConf.save(OmegaConf.create(get_config("rft.framewise").to_dict()), path)
            original = os.getcwd()
            try:
                os.chdir(directory)
                config = runtime.load_training_config(self.arguments(config="", config_path=str(path)))
            finally:
                os.chdir(original)
            self.assertEqual(config.num_frame_per_block, 1)

    def test_logdir_defaults_follow_trainer_for_registry_and_yaml(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"test fixture; never loaded")
            for variant, logdir in (("rft.framewise", "logs/stage1"), ("dmd.framewise", "logs/dmd")):
                path = Path(directory) / "custom.yaml"
                OmegaConf.save(OmegaConf.create(get_config(variant).to_dict()), path)
                for source in ({"config": variant}, {"config": "", "config_path": str(path)}):
                    with self.subTest(variant=variant, source=source):
                        args = self.arguments(**source, generator_ckpt=str(checkpoint), logdir=None)
                        config = runtime.load_training_config(args)
                        self.assertEqual(config.logdir, logdir)
                        self.assertEqual(config.wandb_save_dir, logdir)

    def test_explicit_empty_logdir_is_preserved(self):
        config = self.runtime().load_training_config(self.arguments(config="rft.framewise", logdir=""))
        self.assertEqual(config.logdir, "")
        self.assertEqual(config.wandb_save_dir, "")

    def test_runtime_dump_does_not_write_secrets(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"RANK": "0"}):
            runtime.dump_runtime_config(OmegaConf.create({"wandb_key": "test-only", "seed": 3}), directory)
            self.assertEqual(OmegaConf.load(Path(directory) / "runtime_config.yaml"), {"seed": 3})

    def test_single_training_entrypoint_uses_validated_configuration(self):
        text = (ROOT / "train.py").read_text()
        self.assertIn("load_training_config", text)
        self.assertIn('"--data_path"', text)
        self.assertIn('"--generator_ckpt"', text)
        self.assertIn('"--enable-wandb"', text)
        self.assertFalse((ROOT / "train_gft.py").exists())

    def test_trainers_log_only_public_configuration(self):
        for name in ("trainer/rft.py", "trainer/distillation.py"):
            text = (ROOT / name).read_text()
            self.assertIn("config=public_config(config)", text)
            self.assertNotIn("config.wandb_key", text)
            self.assertNotIn("config.wandb_entity", text)

    def test_temporary_files_use_platform_default(self):
        tree = ast.parse((ROOT / "wan/utils/utils.py").read_text())
        fixed_root = "/" + "tmp"
        self.assertFalse(any(isinstance(node, ast.Constant) and node.value == fixed_root
                             for node in ast.walk(tree)))

    def test_training_script_is_portable(self):
        path = ROOT / "scripts/train.sh"
        self.assertTrue(path.is_file(), "Missing portable launcher")
        text = path.read_text()
        self.assertNotIn("conda activate", text)
        self.assertNotIn("WANDB_API_KEY=", text)
        self.assertIn('"$@"', text)

    def test_merge_module_has_no_top_level_execution_or_delete_calls(self):
        tree = ast.parse((ROOT / "utils/merge_and_get_clean.py").read_text())
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and node.func.attr in {"rmtree", "unlink", "remove"} for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) for node in tree.body))

    def test_merge_cli_preserves_inputs_and_extracts_clean_latent(self):
        path = ROOT / "utils/merge_and_get_clean.py"
        tree = ast.parse(path.read_text())
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body),
                        "Data conversion must expose an explicit CLI")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            original = np.arange(12, dtype=np.float16).reshape(3, 1, 1, 2, 2)
            env = lmdb.open(str(source), map_size=1 << 20)
            with env.begin(write=True) as txn:
                txn.put(b"latents_shape", b"1 3 1 1 2 2")
                txn.put(b"prompts_shape", b"1")
                txn.put(b"latents_0_data", original.tobytes())
                txn.put(b"prompts_0_data", b"A test scene")
            env.close()
            before = (source / "data.mdb").read_bytes()
            dest = Path(directory) / "clean"
            result = subprocess.run([sys.executable, "-B", str(path), "--sources", str(source),
                                     "--output", str(dest)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((source / "data.mdb").read_bytes(), before)
            env = lmdb.open(str(dest), readonly=True, lock=False)
            with env.begin() as txn:
                self.assertEqual(txn.get(b"latents_shape"), b"1 1 1 1 2 2")
                self.assertEqual(txn.get(b"prompts_0_data"), b"A test scene")
                np.testing.assert_array_equal(np.frombuffer(txn.get(b"latents_0_data"), dtype=np.float16), original[-1].ravel())
            env.close()
            repeated = subprocess.run([sys.executable, "-B", str(path), "--sources", str(source),
                                       "--output", str(dest)], capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((source / "data.mdb").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

"""Validate public entry points without initializing CUDA or loading weights."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.registry import get_config


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.checkpoint = self.directory / "model.pt"
        self.checkpoint.write_bytes(b"test fixture; never loaded")
        self.events = []
        self.trainer = SimpleNamespace(train=Mock(side_effect=lambda: self.events.append("train")))
        self.factory = Mock(return_value=self.trainer)
        self.rft_factory = Mock(return_value=self.trainer)
        torch = ModuleType("torch")
        dist = ModuleType("torch.distributed")
        dist.is_initialized = lambda: True
        dist.destroy_process_group = lambda: self.events.append("destroy")
        torch.distributed = dist
        wandb = ModuleType("wandb")
        wandb.finish = lambda: self.events.append("finish")
        trainers = ModuleType("trainer")
        trainers.ScoreDistillationTrainer = self.factory
        rft = ModuleType("trainer.rft")
        rft.Trainer = self.rft_factory
        trainers.rft = rft
        self.modules = {"torch": torch, "torch.distributed": dist, "wandb": wandb,
                        "trainer": trainers, "trainer.rft": rft}

    def invoke(self, variant="dmd.framewise", extra=(), yaml_overrides=None, include_logdir=True):
        spec = importlib.util.spec_from_file_location("public_entry", ROOT / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        argv = ["train.py", "--data_path", str(ROOT / "prompts/demos.txt")]
        if yaml_overrides is None:
            argv += ["--config", variant]
        else:
            config = OmegaConf.merge(OmegaConf.create(get_config(variant).to_dict()), yaml_overrides)
            path = self.directory / "custom.yaml"
            OmegaConf.save(config, path)
            argv += ["--config_path", str(path)]
        if include_logdir:
            argv += ["--logdir", str(self.directory / "logs")]
        if variant.startswith("dmd."):
            argv += ["--generator_ckpt", str(self.checkpoint)]
        with patch.dict(sys.modules, self.modules), patch.object(sys, "argv", argv + list(extra)), \
                patch.dict(os.environ, {"RANK": "0"}), contextlib.redirect_stderr(io.StringIO()):
            module.main()

    def test_registered_variants_dispatch_to_matching_trainer(self):
        for variant in ("rft.framewise", "rft.chunkwise", "dmd.framewise", "dmd.chunkwise"):
            with self.subTest(variant=variant):
                self.factory.reset_mock()
                self.rft_factory.reset_mock()
                self.invoke(variant=variant)
                expected, other = ((self.rft_factory, self.factory) if variant.startswith("rft.")
                                   else (self.factory, self.rft_factory))
                expected.assert_called_once()
                other.assert_not_called()
                self.assertEqual(expected.call_args.args[0].config_name, variant)

    def test_dmd_records_overrides_and_cleans_up(self):
        self.invoke(extra=["--no_visualize"])
        config = self.factory.call_args.args[0]
        self.assertTrue(config.no_visualize)
        self.assertTrue(config.disable_wandb)
        self.assertEqual(config.generator_ckpt, str(self.checkpoint))
        saved = OmegaConf.load(self.directory / "logs/runtime_config.yaml")
        self.assertTrue(saved.no_visualize)
        self.assertEqual(self.events, ["train", "destroy", "finish"])

    def test_stage1_uses_shared_configuration(self):
        self.invoke(variant="rft.framewise")
        config = self.rft_factory.call_args.args[0]
        self.assertEqual(config.perceptual_type, "dit")
        self.assertTrue(config.disable_wandb)
        self.assertEqual(self.events, ["train", "finish"])

    def test_yaml_dispatches_by_trainer(self):
        for variant in ("rft.framewise", "dmd.framewise"):
            with self.subTest(variant=variant):
                self.factory.reset_mock()
                self.rft_factory.reset_mock()
                self.invoke(variant=variant, yaml_overrides={})
                expected, other = ((self.rft_factory, self.factory) if variant.startswith("rft.")
                                   else (self.factory, self.rft_factory))
                expected.assert_called_once()
                other.assert_not_called()
                self.assertEqual(expected.call_args.args[0].config_name, "custom")

    def test_invalid_stage1_supervision_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(variant="rft.framewise", yaml_overrides={"perceptual_type": "lpips"})
        self.assertEqual(error.exception.code, 2)
        self.factory.assert_not_called()
        self.rft_factory.assert_not_called()
        self.assertFalse((self.directory / "logs/runtime_config.yaml").exists())

    def test_unknown_trainer_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(variant="rft.framewise", yaml_overrides={"trainer": "unknown"})
        self.assertEqual(error.exception.code, 2)
        self.factory.assert_not_called()
        self.rft_factory.assert_not_called()

    def test_old_stage1_trainer_names_are_rejected(self):
        for trainer in ("gft_no_cfg", "rft_no_cfg"):
            with self.subTest(trainer=trainer):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(variant="rft.framewise", yaml_overrides={"trainer": trainer})
                self.assertEqual(error.exception.code, 2)
                self.factory.assert_not_called()
                self.rft_factory.assert_not_called()

    def test_default_logdir_reaches_trainer_and_config_dump(self):
        for variant, expected_dir in (("rft.framewise", "logs/stage1"), ("dmd.framewise", "logs/dmd")):
            with self.subTest(variant=variant), patch("utils.runtime.dump_runtime_config") as dump:
                self.invoke(variant=variant, include_logdir=False)
                factory = self.rft_factory if variant.startswith("rft.") else self.factory
                config = factory.call_args.args[0]
                self.assertEqual(config.logdir, expected_dir)
                self.assertEqual(config.wandb_save_dir, expected_dir)
                dump.assert_called_once_with(config, expected_dir)

    def test_explicit_log_directories_override_defaults(self):
        for variant in ("rft.framewise", "dmd.framewise"):
            with self.subTest(variant=variant):
                self.invoke(variant=variant, extra=["--wandb-save-dir", str(self.directory / "wandb")])
                saved = OmegaConf.load(self.directory / "logs/runtime_config.yaml")
                self.assertEqual(saved.logdir, str(self.directory / "logs"))
                self.assertEqual(saved.wandb_save_dir, str(self.directory / "wandb"))

    def test_removed_training_limit_is_rejected_before_trainer(self):
        for variant in ("rft.framewise", "dmd.framewise"):
            with self.subTest(variant=variant):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(variant=variant, extra=["--max_train_steps", "7"])
                self.assertEqual(error.exception.code, 2)
                self.factory.assert_not_called()
                self.rft_factory.assert_not_called()

    def test_training_errors_propagate(self):
        self.trainer.train.side_effect = RuntimeError("training failed")
        for variant in ("rft.framewise", "dmd.framewise"):
            with self.subTest(variant=variant), self.assertRaisesRegex(RuntimeError, "training failed"):
                self.invoke(variant=variant)
        self.assertEqual(self.events, [])

    def test_wandb_requires_explicit_enable(self):
        self.invoke(extra=["--enable-wandb"])
        self.assertFalse(self.factory.call_args.args[0].disable_wandb)

    def test_help_works_without_loading_gpu_models(self):
        code = (
            "import runpy, sys; "
            "sys.modules.update({name: None for name in ('torch', 'trainer', 'wandb')}); "
            "sys.argv = [sys.argv[1], '--help']; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "train.py")],
                                cwd=self.directory, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--generator_ckpt", result.stdout)
        self.assertIn("rft.framewise", result.stdout)
        self.assertIn("dmd.framewise", result.stdout)

    def test_launcher_handles_foreign_cwd_spaces_and_failure(self):
        fake = self.directory / "python stub"
        fake.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > "$RECORD"\nexit "${STUB_EXIT:-0}"\n')
        fake.chmod(0o700)
        record = self.directory / "arguments.txt"
        env = dict(os.environ, PYTHON=str(fake), RECORD=str(record), GPUS_PER_NODE="2",
                   NNODES="1", NODE_RANK="0", LOGDIR="logs/test")
        for variant in ("rft.framewise", "rft.chunkwise", "dmd.framewise", "dmd.chunkwise"):
            with self.subTest(variant=variant):
                command = ["sh", str(ROOT / "scripts/train.sh"), variant,
                           "--generator_ckpt", "checkpoints/stage one.pt", "--no_visualize"]
                result = subprocess.run(command, cwd=self.directory, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = record.read_text().splitlines()
                self.assertEqual(arguments[0], str(ROOT))
                self.assertIn("checkpoints/stage one.pt", arguments)
                self.assertIn("--no_visualize", arguments)
                self.assertIn("torch.distributed.run", arguments)
                self.assertIn("2", arguments)
                self.assertEqual([arg for arg in arguments if arg.endswith(".py")], ["train.py"])
                self.assertEqual(arguments[arguments.index("--config") + 1], variant)
        env["STUB_EXIT"] = "17"
        result = subprocess.run(command, cwd=self.directory, env=env, capture_output=True)
        self.assertEqual(result.returncode, 17)


if __name__ == "__main__":
    unittest.main()

# Modified for the RepForcing source release; see NOTICE.md.
"""Exercise the actual unbounded DMD training loop with mocked GPU operations."""
import ast
import contextlib
import io
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


class Scalar:
    def __init__(self, value=1.0):
        self.value = value

    def mean(self):
        return self

    def item(self):
        return self.value


class StopUnboundedLoop(Exception):
    """Interrupt continuous training from the test data iterator."""


def load_trainer_methods():
    tree = ast.parse((ROOT / "trainer/distillation.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "train"]
    namespace = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()), is_tensor=lambda value: False),
        "dist": SimpleNamespace(get_rank=lambda: 0),
        "merge_scalar_logs": lambda logs: logs[0],
        "EMA_FSDP": Mock(return_value=SimpleNamespace(update=Mock())),
        "wandb": SimpleNamespace(log=Mock()), "time": time,
        "logging": SimpleNamespace(info=Mock()), "gc": SimpleNamespace(collect=Mock()),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / "trainer/distillation.py"), "exec"), namespace)
    return namespace["Trainer"], namespace


class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        self.trainer_type, self.namespace = load_trainer_methods()

    def fake_trainer(self, iterations=3, interval=2, no_save=False, no_visualize=False, rank=0, start=0):
        config = SimpleNamespace(
            dfake_gen_update_ratio=5, ema_start_step=200, ema_weight=0,
            no_save=no_save, no_visualize=no_visualize, log_iters=interval, gc_interval=100000,
        )
        trainer = SimpleNamespace(
            config=config, step=start, is_main_process=rank == 0,
            disable_wandb=True, grad_accum=2,
            generator_optimizer=Mock(), critic_optimizer=Mock(), generator_ema=None,
            model=SimpleNamespace(generator=Mock(), fake_score=Mock()),
            max_grad_norm_generator=1.0, max_grad_norm_critic=1.0, previous_time=None,
        )

        def batches():
            while trainer.step < start + iterations:
                yield {"prompts": ["A test scene"]}
            raise StopUnboundedLoop()

        trainer.dataloader = batches()
        trainer.model.generator.clip_grad_norm_.return_value = Scalar()
        trainer.model.fake_score.clip_grad_norm_.return_value = Scalar()
        trainer.fwdbwd_one_step = Mock(side_effect=lambda batch, train_generator: {
            "generator_loss" if train_generator else "critic_loss": Scalar(),
        })
        trainer.saved_steps = []
        trainer.visualized_steps = []
        trainer.save = Mock(side_effect=lambda: trainer.saved_steps.append(trainer.step))
        trainer._visualize_wandb = Mock(side_effect=lambda: trainer.visualized_steps.append(trainer.step))
        self.namespace["dist"].get_rank = lambda: rank
        return trainer

    def run_loop(self, trainer):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(StopUnboundedLoop):
            self.trainer_type.train(trainer)

    def test_update_counts_and_periodic_save(self):
        trainer = self.fake_trainer(iterations=1200, interval=50)
        self.run_loop(trainer)
        self.assertEqual(trainer.step, 1200)
        self.assertEqual(trainer.critic_optimizer.step.call_count, 1200)
        self.assertEqual(trainer.generator_optimizer.step.call_count, 240)
        self.assertEqual(trainer.fwdbwd_one_step.call_count, (1200 + 240) * 2)
        self.assertEqual(trainer.saved_steps, list(range(50, 1201, 50)))
        self.assertEqual(trainer.visualized_steps, trainer.saved_steps)

    def test_interruption_does_not_add_nonperiodic_save(self):
        trainer = self.fake_trainer()
        self.run_loop(trainer)
        self.assertEqual(trainer.step, 3)
        self.assertEqual(trainer.saved_steps, [2])
        self.assertEqual(trainer.visualized_steps, [2])
        self.assertEqual(trainer.generator_optimizer.step.call_count, 1)

    def test_no_save(self):
        trainer = self.fake_trainer(no_save=True, interval=0)
        self.run_loop(trainer)
        self.assertEqual(trainer.saved_steps, [])
        self.assertEqual(trainer.visualized_steps, [])

    def test_no_visualize_preserves_periodic_save(self):
        trainer = self.fake_trainer(no_visualize=True)
        self.run_loop(trainer)
        self.assertEqual(trainer.saved_steps, [2])
        self.assertEqual(trainer.visualized_steps, [])

    def test_non_main_rank_also_saves(self):
        trainer = self.fake_trainer(rank=1)
        self.run_loop(trainer)
        self.assertEqual(trainer.step, 3)
        self.assertEqual(trainer.saved_steps, [2])

    def test_nonzero_start_preserves_schedule(self):
        trainer = self.fake_trainer(start=3)
        self.run_loop(trainer)
        self.assertEqual(trainer.step, 6)
        self.assertEqual(trainer.critic_optimizer.step.call_count, 3)
        self.assertEqual(trainer.generator_optimizer.step.call_count, 1)
        self.assertEqual(trainer.saved_steps, [4, 6])

    def test_save_failure_propagates(self):
        trainer = self.fake_trainer()
        trainer.save.side_effect = OSError("checkpoint write failed")
        with self.assertRaisesRegex(OSError, "checkpoint write failed"):
            self.run_loop(trainer)


if __name__ == "__main__":
    unittest.main()

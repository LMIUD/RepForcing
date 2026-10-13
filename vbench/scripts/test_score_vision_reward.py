# Modified for the RepForcing source release; see NOTICE.md.
"""Test scoring, input mapping, and resume validation without loading models."""

import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import score_vision_reward as scoring


class VisionRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions, cls.weights, cls.instruction, cls.source = scoring.load_protocol()

    def record(self, path="fixtures/video.mp4", instruction="yes"):
        answers = ["yes"] * 29
        return {
            "video_path": path,
            "prompt": "test prompt",
            "answers": answers,
            "instruction_answer": instruction,
            **scoring.aggregate_answers(answers, self.weights, instruction),
        }

    def jsonl_path(self, content):
        path = MagicMock()
        path.exists.return_value = True
        path.open.return_value = io.StringIO(content)
        return path

    def test_official_protocol(self):
        self.assertEqual(len(self.questions), 29)
        self.assertEqual(len(self.weights), 29)
        self.assertEqual(self.instruction.rstrip("\r\n"), scoring.IF_QUESTION)
        self.assertNotIn(self.instruction, self.questions)
        with (scoring.ASSET_DIR / "questions.txt").open() as handle:
            self.assertEqual(self.questions, handle.readlines())
        self.assertEqual(self.source["commit"], scoring.OFFICIAL_COMMIT)

    def test_mean_is_not_weight_normalized_sum(self):
        result = scoring.aggregate_answers(["yes"] * 29, self.weights, "yes")
        self.assertAlmostEqual(result["vision_reward"], sum(self.weights) / 29)
        self.assertNotAlmostEqual(result["vision_reward"], 1.0)

    def test_official_binary_mapping(self):
        for answer, expected in [("yes", 1), ("no", -1), ("Yes", -1), (" yes", -1), ("maybe", -1)]:
            self.assertEqual(scoring.encode_answer(answer), expected)

    def test_if_is_independent(self):
        positive = scoring.aggregate_answers(["yes"] * 29, self.weights, "yes")
        negative = scoring.aggregate_answers(["yes"] * 29, self.weights, "no")
        self.assertEqual(positive["vision_reward"], negative["vision_reward"])
        self.assertEqual(positive["instruction_following"], 1)
        self.assertEqual(negative["instruction_following"], -1)

    def test_mixed_answers(self):
        answers = ["yes", "no"] * 14 + ["yes"]
        values = np.array([1 if answer == "yes" else -1 for answer in answers])
        expected = np.mean(values * np.asarray(self.weights)).item()
        self.assertEqual(scoring.aggregate_answers(answers, self.weights, "no")["vision_reward"], expected)

    def test_unexpected_answers_are_flagged(self):
        result = scoring.aggregate_answers(["maybe"] + ["yes"] * 28, self.weights, "Yes")
        self.assertEqual(result["unexpected_answers"], [0])
        self.assertTrue(result["unexpected_instruction_answer"])

    def test_answer_count_mismatch(self):
        with self.assertRaises(ValueError):
            scoring.aggregate_answers(["yes"], self.weights, "yes")

    def test_partial_and_complete_summary(self):
        records = {f"fixtures/{i}.mp4": self.record(f"fixtures/{i}.mp4") for i in range(100)}
        full = scoring.make_summary(records, 100)
        self.assertEqual(full["status"], "complete")
        self.assertEqual(full["num_failed_or_pending"], 0)
        self.assertAlmostEqual(full["vision_reward_pct"], 100 * sum(self.weights) / 29)
        self.assertEqual(full["instruction_following_pct"], 100)
        subset = scoring.make_summary({"one": self.record()}, 1)
        self.assertEqual(subset["status"], "partial")
        self.assertTrue(subset["complete_requested_subset"])
        pending = scoring.make_summary({"one": self.record()}, 100)
        self.assertEqual(pending["num_failed_or_pending"], 99)
        self.assertEqual(pending["status"], "partial")

    def test_empty_summary_is_not_zero_score(self):
        result = scoring.make_summary({}, 100)
        self.assertIsNone(result["vision_reward"])
        self.assertIsNone(result["instruction_following"])
        self.assertEqual(result["status"], "partial")

    def test_read_valid_completed_record(self):
        record = self.record()
        path = self.jsonl_path(json.dumps(record) + "\n")
        items = [{"video_path": record["video_path"], "prompt": record["prompt"]}]
        self.assertEqual(scoring.read_records(path, items, self.weights), {record["video_path"]: record})

    def test_reject_duplicate_or_truncated_records(self):
        record = self.record()
        items = [{"video_path": record["video_path"], "prompt": record["prompt"]}]
        for content in [(json.dumps(record) + "\n") * 2, json.dumps(record)]:
            with self.assertRaises(ValueError):
                scoring.read_records(self.jsonl_path(content), items, self.weights)

    def test_reject_changed_scores_or_prompts(self):
        record = self.record()
        items = [{"video_path": record["video_path"], "prompt": record["prompt"]}]
        for changes in [{"vision_reward": 100}, {"prompt": "different prompt"}]:
            changed = dict(record, **changes)
            with self.assertRaises(ValueError):
                scoring.read_records(self.jsonl_path(json.dumps(changed) + "\n"), items, self.weights)

    def test_prompt_mapping_preserves_full_text(self):
        prompts = [f"Video {i:03d}: " + "a long description " * 10 for i in range(100)]
        prompts_path = MagicMock()
        prompts_path.read_text.return_value = "\n".join(prompts)
        directory = Path("fixtures/videos")
        files = [directory / (p[:100] + ".mp4") for p in prompts]
        with patch.object(Path, "is_dir", return_value=True), patch.object(Path, "is_file", return_value=True), patch.object(Path, "iterdir", return_value=iter(files)):
            items = scoring.resolve_inputs(directory, prompts_path)
        self.assertEqual(items[0]["prompt"], prompts[0].rstrip())
        self.assertGreater(len(items[0]["prompt"]), 100)
        self.assertEqual(len(items), 100)

    def test_reject_truncated_filename_collision(self):
        prompts_path = MagicMock()
        prompts_path.read_text.return_value = "\n".join("a" * 100 + str(i) for i in range(100))
        with self.assertRaises(ValueError):
            scoring.resolve_inputs(Path("fixtures/videos"), prompts_path)


if __name__ == "__main__":
    unittest.main()

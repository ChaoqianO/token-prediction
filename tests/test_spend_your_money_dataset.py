from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    SweBenchTaskMetadata,
    build_spend_your_money_dataset,
)


class SpendYourMoneyDatasetTests(unittest.TestCase):
    def test_one_model_condition_builds_task_launch_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.csv"
            path.write_text(
                "problem_id,gpt52_gt_input_token_avg,gpt52_gt_output_token_avg,"
                "gpt52_predicted_avg_input,gpt52_predicted_avg_output\n"
                "org__repo-1,100.25,20.25,90,15\n"
                "org__repo-2,200,50,180,35\n",
                encoding="utf-8",
            )
            metadata = {
                "org__repo-1": SweBenchTaskMetadata(
                    "org__repo-1",
                    "org/repo",
                    "Fix this bug.\n```python\nraise Bug()\n```",
                ),
                "org__repo-2": SweBenchTaskMetadata(
                    "org__repo-2", "org/repo", "Handle the second regression."
                ),
            }
            imported = build_spend_your_money_dataset(
                path,
                metadata,
                model_key="gpt52",
                model_id="gpt-5.2",
            )

        dataset_slice = imported.dataset.select(
            PredictionPosition.TASK_LAUNCH,
            PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
        )
        self.assertEqual(imported.task_count, 2)
        self.assertEqual({row.label for row in dataset_slice.rows}, {120, 250})
        first = next(
            row for row in dataset_slice.rows if row.point.task_id.endswith("org__repo-1")
        )
        self.assertEqual(first.point.features["repo_id"], "org/repo")
        self.assertEqual(first.point.features["model_id"], "gpt-5.2")
        self.assertEqual(first.point.features["task_code_fence_count"], 1)
        self.assertEqual(
            first.point.features["llm_self_estimated_total_tokens"], 105.0
        )
        self.assertNotIn("gpt52_predicted_avg_input", first.point.features)

    def test_missing_task_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.csv"
            path.write_text(
                "problem_id,gpt52_gt_input_token_avg,gpt52_gt_output_token_avg,"
                "gpt52_predicted_avg_input,gpt52_predicted_avg_output\n"
                "missing-task,1,2,1,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing SWE-bench metadata"):
                build_spend_your_money_dataset(path, {}, model_key="gpt52")


if __name__ == "__main__":
    unittest.main()

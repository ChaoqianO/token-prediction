from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from token_prediction.config import load_config
from token_prediction.contracts import Observable, SourceCapabilities, SourceDescriptor
from token_prediction.experiment import validate_ablation_specs


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_every_mvp_declaration_is_parsed(self) -> None:
        config = load_config(ROOT / "configs" / "mvp.toml")
        self.assertEqual(config.folds, 5)
        self.assertEqual(len(config.candidates), 2)
        self.assertEqual(len(config.experiment_specs()), 3)
        self.assertTrue(all(spec.required_features for spec in config.experiment_specs()))
        codex = load_config(ROOT / "configs" / "codex_task_mvp.toml")
        self.assertEqual(codex.experiment_specs()[0].position.value, "task_launch")
        lightgbm = load_config(ROOT / "configs" / "lightgbm_mvp.toml")
        self.assertEqual(len(lightgbm.candidates), 5)
        self.assertEqual(lightgbm.experiment_specs()[0].position.value, "task_update")
        validate_ablation_specs(lightgbm.candidates)

    def test_unknown_key_is_rejected_instead_of_becoming_dead_config(self) -> None:
        source = (ROOT / "configs" / "mvp.toml").read_text(encoding="utf-8")
        broken = source.replace("schema_version = 1", "schema_version = 1\nunused_switch = true")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.toml"
            path.write_text(broken, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown top-level"):
                load_config(path)

    def test_schema_v2_requires_and_parses_source_capability_contract(self) -> None:
        source = (ROOT / "configs" / "mvp.toml").read_text(encoding="utf-8")
        source = source.replace("schema_version = 1", "schema_version = 2", 1)
        source = source.replace(
            'estimator = "empirical_quantile"',
            'estimator = "cross_position_deduct"',
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            descriptor = SourceDescriptor(
                source_id="fixture_canonical_v2",
                revision="fixture-revision-1",
                manifest_path="workspace/manifests/fixture.json",
                manifest_sha256="0" * 64,
                capabilities=SourceCapabilities(
                    source_id="fixture_canonical_v2",
                    source="test",
                    observables=frozenset(
                        {
                            Observable.ATTEMPT_USAGE,
                            Observable.REQUEST_BOUNDARIES,
                            Observable.REQUEST_LOCAL_COUNT,
                            Observable.TASK_TERMINATION,
                        }
                    ),
                ),
            )
            descriptor_path = root / "configs" / "source_descriptors" / "fixture.json"
            descriptor_path.parent.mkdir(parents=True)
            descriptor_path.write_text(
                json.dumps(descriptor.to_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
            source = source.replace(
                'source = "canonical_jsonl"',
                "\n".join(
                    (
                        'source = "canonical_jsonl"',
                        'descriptor_path = "configs/source_descriptors/fixture.json"',
                        f'descriptor_sha256 = "{descriptor_sha}"',
                        'canonical_manifest_path = "manifests/canonical.json"',
                        f'canonical_manifest_sha256 = "{"1" * 64}"',
                    )
                ),
                1,
            )
            source = source.replace(
                "[candidates.params]\nalpha = 0.10",
                "\n".join(
                    (
                        "[candidates.params]",
                        "alpha = 0.10",
                        "",
                        "[candidates.initializer_params]",
                        "alpha = 0.20",
                        "minimum = 3",
                        "",
                        "[candidates.graph]",
                        'initializer = "empirical_quantile"',
                        'updater = "cross_position_deduct"',
                        'lifecycle_schema = "task_lifecycle_v1"',
                        'seed_policy = "inner_oof_repaired_quantile_mean_v1"',
                        'inner_split_policy = "five_fold_rotating_holdout_v1"',
                    )
                ),
                1,
            )
            path = root / "configs" / "v2.toml"
            path.write_text(source, encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.schema_version, 2)
        self.assertIsNotNone(config.source_descriptor)
        assert config.source_descriptor is not None
        self.assertEqual(config.source_descriptor.source_id, "fixture_canonical_v2")
        self.assertEqual(len(config.source_descriptor.descriptor_hash), 64)
        self.assertTrue(config.candidates[0].graph.is_lifecycle)
        self.assertEqual(
            config.candidates[0].initializer_params,
            {"alpha": 0.2, "minimum": 3},
        )
        self.assertEqual(
            config.experiment_specs()[0].candidates[0].content_hash,
            config.candidates[0].content_hash,
        )

    def test_schema_v1_rejects_partial_source_descriptor(self) -> None:
        source = (ROOT / "configs" / "mvp.toml").read_text(encoding="utf-8")
        broken = source.replace(
            'source = "canonical_jsonl"',
            'source = "canonical_jsonl"\ndescriptor_path = "descriptor.json"',
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.toml"
            path.write_text(broken, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot declare v2 source manifests"):
                load_config(path)

    def test_schema_v2_rejects_raw_path_whitespace(self) -> None:
        source = (ROOT / "configs" / "mvp.toml").read_text(encoding="utf-8")
        source = source.replace("schema_version = 1", "schema_version = 2", 1)
        source = source.replace(
            'source = "canonical_jsonl"',
            "\n".join(
                (
                    'source = "canonical_jsonl"',
                    'descriptor_path = " configs/source_descriptors/fixture.json"',
                    f'descriptor_sha256 = "{"0" * 64}"',
                    'canonical_manifest_path = "manifests/canonical.json"',
                    f'canonical_manifest_sha256 = "{"1" * 64}"',
                )
            ),
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "configs" / "v2.toml"
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical relative POSIX"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

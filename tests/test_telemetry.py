from __future__ import annotations

import unittest

from token_prediction.contracts import Observable, SourceCapabilities
from token_prediction.telemetry import (
    TelemetryCapabilityError,
    TelemetrySurface,
    decide_telemetry_surface,
    require_telemetry_surface,
)


class TelemetryGateTests(unittest.TestCase):
    def test_each_surface_uses_an_explicit_independent_capability_contract(self) -> None:
        capabilities = SourceCapabilities(
            "source",
            frozenset(
                {
                    Observable.ATTEMPT_USAGE,
                    Observable.REQUEST_BOUNDARIES,
                    Observable.TASK_TERMINATION,
                }
            ),
        )
        lifecycle = decide_telemetry_surface(
            capabilities,
            TelemetrySurface.TASK_LIFECYCLE,
        )
        shadow = decide_telemetry_surface(
            capabilities,
            TelemetrySurface.ONLINE_SHADOW,
        )
        call_update = decide_telemetry_surface(
            capabilities,
            TelemetrySurface.CALL_UPDATE,
        )
        self.assertTrue(lifecycle.available)
        self.assertTrue(shadow.available)
        self.assertFalse(call_update.available)
        self.assertEqual(call_update.missing_observables, ("output_deltas",))
        self.assertEqual(
            call_update.reason,
            "missing_observables:output_deltas",
        )
        self.assertNotEqual(lifecycle.decision_id, shadow.decision_id)

    def test_g3_surfaces_do_not_infer_hidden_or_logprob_telemetry(self) -> None:
        capabilities = SourceCapabilities(
            "source",
            frozenset({Observable.OUTPUT_DELTAS}),
        )
        self.assertTrue(
            decide_telemetry_surface(
                capabilities,
                TelemetrySurface.G3_GENERATION_PROGRESS,
            ).available
        )
        entropy = decide_telemetry_surface(
            capabilities,
            TelemetrySurface.G3_ENTROPY_STOP,
        )
        hidden = decide_telemetry_surface(
            capabilities,
            TelemetrySurface.G3_HIDDEN_STATE,
        )
        self.assertEqual(entropy.missing_observables, ("logprobs",))
        self.assertEqual(hidden.missing_observables, ("hidden_state",))

    def test_decision_identity_is_order_independent_and_requirement_is_enforced(
        self,
    ) -> None:
        first = SourceCapabilities(
            "source",
            frozenset(
                {
                    Observable.REQUEST_BOUNDARIES,
                    Observable.ATTEMPT_USAGE,
                }
            ),
        )
        second = SourceCapabilities(
            "source",
            frozenset(
                {
                    Observable.ATTEMPT_USAGE,
                    Observable.REQUEST_BOUNDARIES,
                }
            ),
        )
        self.assertEqual(
            decide_telemetry_surface(first, "call_pre").decision_id,
            decide_telemetry_surface(second, "call_pre").decision_id,
        )
        missing = SourceCapabilities(
            "source",
            frozenset({Observable.REQUEST_BOUNDARIES}),
        )
        with self.assertRaises(TelemetryCapabilityError) as caught:
            require_telemetry_surface(missing, TelemetrySurface.ONLINE_SHADOW)
        self.assertEqual(
            caught.exception.decision.missing_observables,
            ("attempt_usage",),
        )


if __name__ == "__main__":
    unittest.main()

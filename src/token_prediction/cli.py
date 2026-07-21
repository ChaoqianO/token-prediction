from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from token_prediction.collection import (
    CodexAuthState,
    CodexCLI,
    CodexTurnMetadata,
    CodexTurnReader,
    write_canonical_jsonl,
)
from token_prediction.config import load_config
from token_prediction.pipeline import inspect_replay, run_configured_experiments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tp", description="Token prediction research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    auth = commands.add_parser("auth", help="Delegate authentication to a supported source")
    auth_sub = auth.add_subparsers(dest="backend", required=True)
    codex = auth_sub.add_parser("codex", help="Official Codex CLI authentication")
    codex.add_argument("action", choices=("status", "login", "logout"))
    codex.add_argument("--executable", default=None)

    doctor = commands.add_parser(
        "doctor", help="Inspect a trajectory source without spending tokens"
    )
    doctor.add_argument("backend", choices=("codex",))
    doctor.add_argument("--executable", default=None)

    replay = commands.add_parser("replay", help="Inspect a canonical JSONL fixture")
    replay.add_argument("--events", required=True, type=Path)

    experiment = commands.add_parser(
        "experiment", help="Run configured baselines on frozen canonical trajectories"
    )
    experiment.add_argument("--config", required=True, type=Path)
    experiment.add_argument("--events", required=True, nargs="+", type=Path)
    experiment.add_argument("--output", default=None, type=Path)

    ingest = commands.add_parser("ingest", help="Normalize a preserved raw run")
    ingest_sub = ingest.add_subparsers(dest="source", required=True)
    codex_turn = ingest_sub.add_parser(
        "codex-turn", help="Normalize one saved codex exec --json turn"
    )
    codex_turn.add_argument("--raw", required=True, type=Path)
    codex_turn.add_argument("--output", required=True, type=Path)
    codex_turn.add_argument("--task-id", required=True)
    codex_turn.add_argument("--started-at", required=True)
    codex_turn.add_argument("--finished-at", required=True)
    codex_turn.add_argument("--task-tokens", type=int, default=None)
    codex_turn.add_argument("--task-hash", default=None)
    codex_turn.add_argument("--run-id", default=None)
    codex_turn.add_argument("--condition-id", default=None)
    codex_turn.add_argument("--model-id", default=None)
    codex_turn.add_argument("--resolved-model-id", default=None)
    codex_turn.add_argument("--reasoning-effort", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        print(json.dumps(inspect_replay(args.events), ensure_ascii=False, indent=2))
        return 0

    if args.command == "experiment":
        summary = run_configured_experiments(
            load_config(args.config), args.events, output_dir=args.output
        )
        payload = asdict(summary)
        payload["output_dir"] = str(summary.output_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "ingest":
        trajectory = CodexTurnReader().read(
            args.raw,
            CodexTurnMetadata(
                task_id=args.task_id,
                started_at=args.started_at,
                finished_at=args.finished_at,
                task_tokens=args.task_tokens,
                task_hash=args.task_hash,
                run_id=args.run_id,
                condition_id=args.condition_id,
                model_id=args.model_id,
                resolved_model_id=args.resolved_model_id,
                reasoning_effort=args.reasoning_effort,
            ),
        )
        output = write_canonical_jsonl(args.output, trajectory.events)
        print(
            json.dumps(
                {
                    "source": "codex_exec_jsonl_turn_v1",
                    "trajectory_id": trajectory.trajectory_id,
                    "events": len(trajectory.events),
                    "output": str(output),
                    "observables": ["task_usage"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command in {"auth", "doctor"}:
        client = CodexCLI(executable=args.executable)
        if args.command == "doctor":
            status = client.auth_status()
            print(
                json.dumps(
                    {
                        "state": status.state.value,
                        "executable": status.executable,
                        "message": status.message,
                        "live_collection": "disabled",
                        "reason": (
                            "The installed Codex JSONL surface has only been verified for "
                            "turn-level aggregate usage; call/request boundaries are not inferred."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if status.state == CodexAuthState.AUTHENTICATED else 1
        if args.action == "status":
            status = client.auth_status()
        elif args.action == "login":
            status = client.login()
        else:
            status = client.logout()
        print(
            json.dumps(
                {
                    "state": status.state.value,
                    "executable": status.executable,
                    "message": status.message,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.action == "logout":
            return 0 if status.state == CodexAuthState.UNAUTHENTICATED else 1
        return 0 if status.state == CodexAuthState.AUTHENTICATED else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

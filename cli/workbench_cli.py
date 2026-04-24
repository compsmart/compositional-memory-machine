from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web import HHRWebState

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"


def _scenario_path(args) -> Path:
    if args.list_fixtures:
        fixtures = sorted(path.stem for path in SCENARIO_DIR.glob("*.json"))
        for fixture in fixtures:
            print(fixture)
        raise SystemExit(0)
    if args.fixture is not None:
        path = SCENARIO_DIR / f"{args.fixture}.json"
        if not path.exists():
            raise SystemExit(f"Unknown fixture: {args.fixture}")
        return path
    if args.scenario is None:
        raise SystemExit("Provide --scenario or --fixture.")
    return args.scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless MemoryWorkbench scenario runner.")
    parser.add_argument("--scenario", type=Path, help="Path to a JSON scenario file.")
    parser.add_argument("--fixture", help="Name of a checked-in scenario fixture from scenarios/.")
    parser.add_argument("--list-fixtures", action="store_true", help="List checked-in scenario fixtures and exit.")
    parser.add_argument("--output", type=Path, help="Optional path to write the resulting snapshot JSON.")
    args = parser.parse_args()

    scenario_path = _scenario_path(args)
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    state = HHRWebState()
    snapshot = state.load_scenario(scenario)
    body = json.dumps(snapshot, indent=2)
    if args.output is not None:
        args.output.write_text(body + "\n", encoding="utf-8")
    else:
        print(body)


if __name__ == "__main__":
    main()

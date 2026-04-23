from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web import HHRWebState


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless MemoryWorkbench scenario runner.")
    parser.add_argument("--scenario", type=Path, required=True, help="Path to a JSON scenario file.")
    parser.add_argument("--output", type=Path, help="Optional path to write the resulting snapshot JSON.")
    args = parser.parse_args()

    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    state = HHRWebState()
    snapshot = state.load_scenario(scenario)
    body = json.dumps(snapshot, indent=2)
    if args.output is not None:
        args.output.write_text(body + "\n", encoding="utf-8")
    else:
        print(body)


if __name__ == "__main__":
    main()

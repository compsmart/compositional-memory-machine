from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language import NGramLanguageMemory


def _sequence_family(family: int, branch_bits: tuple[int, int, int, int]) -> list[str]:
    b0, b1, b2, b3 = branch_bits
    return [
        f"fam{family}_start",
        f"fam{family}_prefix",
        f"choice0_{b0}",
        f"fam{family}_pivot0",
        f"choice1_{b1}",
        f"fam{family}_pivot1",
        f"choice2_{b2}",
        f"fam{family}_pivot2",
        f"choice3_{b3}",
        f"fam{family}_end",
    ]


def _training_sequences(n_sequences: int) -> list[list[str]]:
    rows: list[list[str]] = []
    family = 0
    while len(rows) < n_sequences:
        for bits in product((0, 1), repeat=4):
            rows.append(_sequence_family(family, bits))
            if len(rows) == n_sequences:
                break
        family += 1
    return rows


def run(
    *,
    dim: int = 4096,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_sequences_values: tuple[int, ...] = (25, 50),
    strategies: tuple[str, ...] = ("greedy_nn", "beam", "top_k_sample"),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for seed in seeds:
        for n_sequences in n_sequences_values:
            sequences = _training_sequences(n_sequences)
            for strategy in strategies:
                model = NGramLanguageMemory(dim=dim, seed=seed)
                for sequence in sequences:
                    model.learn_sequence(sequence)

                seq_hits = 0
                token_hits = 0
                token_total = 0
                for row_idx, sequence in enumerate(sequences):
                    generated = model.generate(
                        sequence[:2],
                        steps=len(sequence) - 2,
                        strategy=strategy,
                        top_k=5,
                        beam_width=3,
                        min_confidence=0.0,
                        rng=None,
                    )
                    target = sequence[2:]
                    seq_hits += int(generated == target)
                    token_hits += sum(int(left == right) for left, right in zip(generated, target))
                    token_total += len(target)

                rows.append(
                    {
                        "dim": float(dim),
                        "seed": float(seed),
                        "n_sequences": float(n_sequences),
                        "strategy": strategy,
                        "seq_em": seq_hits / len(sequences),
                        "tok_acc": token_hits / max(token_total, 1),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sequences", type=int, nargs="+", default=[25, 50])
    args = parser.parse_args()

    for row in run(dim=args.dim, seeds=tuple(args.seeds), n_sequences_values=tuple(args.sequences)):
        print(row)


if __name__ == "__main__":
    main()

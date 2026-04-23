from __future__ import annotations

from hrr.binding import bind, cosine, normalize, unbind
from hrr.encoder import SVOEncoder


def test_binding_unbinding_recovers_filler_direction() -> None:
    encoder = SVOEncoder(dim=2048, seed=11)
    role = encoder.store.get_unitary("role")
    filler = encoder.store.get("filler")

    recovered = normalize(unbind(bind(role, filler), role))

    assert cosine(recovered, filler) > 0.9


def test_svo_encoding_is_stable_for_same_seed() -> None:
    left = SVOEncoder(dim=256, seed=1)
    right = SVOEncoder(dim=256, seed=1)

    assert cosine(left.encode("doctor", "treats", "patient"), right.encode("doctor", "treats", "patient")) > 0.999
    assert cosine(left.encode("doctor", "treats", "patient"), left.encode("chef", "prepares", "meal")) < 0.3


def test_temporal_encoding_changes_representation() -> None:
    encoder = SVOEncoder(dim=256, seed=3)

    observed = encoder.encode_temporal("robot", "location", "lab", time_token="t0", state_token="observed")
    revised = encoder.encode_temporal("robot", "location", "lab", time_token="t1", state_token="revised")

    assert cosine(observed, revised) < 0.99

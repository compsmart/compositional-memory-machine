# Temporal Ordering Frontier

## Configuration

- `dims=[1024, 2048, 4096]`
- `event_counts=[50, 100, 200]`
- `chunk_size=25`
- `seeds=[0, 1, 2]`

## Summary

| strategy | dim | n_events | runs | latest_state_em | pairwise_order_em | balanced_score |
| --- | --- | --- | --- | --- | --- | --- |
| chunked_temporal_roles | 1024 | 50 | 3 | 0.000 | 0.962 | 0.481 |
| chunked_temporal_roles | 1024 | 100 | 3 | 0.000 | 0.908 | 0.454 |
| chunked_temporal_roles | 1024 | 200 | 3 | 0.000 | 0.857 | 0.429 |
| chunked_temporal_roles | 2048 | 50 | 3 | 0.056 | 1.000 | 0.528 |
| chunked_temporal_roles | 2048 | 100 | 3 | 0.000 | 0.996 | 0.498 |
| chunked_temporal_roles | 2048 | 200 | 3 | 0.056 | 0.993 | 0.524 |
| chunked_temporal_roles | 4096 | 50 | 3 | 0.000 | 1.000 | 0.500 |
| chunked_temporal_roles | 4096 | 100 | 3 | 0.000 | 1.000 | 0.500 |
| chunked_temporal_roles | 4096 | 200 | 3 | 0.000 | 1.000 | 0.500 |
| explicit_pair_links | 1024 | 50 | 3 | 0.000 | 0.544 | 0.272 |
| explicit_pair_links | 1024 | 100 | 3 | 0.000 | 0.465 | 0.232 |
| explicit_pair_links | 1024 | 200 | 3 | 0.000 | 0.256 | 0.128 |
| explicit_pair_links | 2048 | 50 | 3 | 0.000 | 0.510 | 0.255 |
| explicit_pair_links | 2048 | 100 | 3 | 0.000 | 0.508 | 0.254 |
| explicit_pair_links | 2048 | 200 | 3 | 0.000 | 0.405 | 0.203 |
| explicit_pair_links | 4096 | 50 | 3 | 0.000 | 0.497 | 0.248 |
| explicit_pair_links | 4096 | 100 | 3 | 0.000 | 0.512 | 0.256 |
| explicit_pair_links | 4096 | 200 | 3 | 0.000 | 0.531 | 0.265 |
| flat_temporal_roles | 1024 | 50 | 3 | 0.056 | 0.856 | 0.456 |
| flat_temporal_roles | 1024 | 100 | 3 | 0.000 | 0.582 | 0.291 |
| flat_temporal_roles | 1024 | 200 | 3 | 0.000 | 0.510 | 0.255 |
| flat_temporal_roles | 2048 | 50 | 3 | 0.000 | 0.955 | 0.477 |
| flat_temporal_roles | 2048 | 100 | 3 | 0.000 | 0.730 | 0.365 |
| flat_temporal_roles | 2048 | 200 | 3 | 0.000 | 0.526 | 0.263 |
| flat_temporal_roles | 4096 | 50 | 3 | 0.000 | 1.000 | 0.500 |
| flat_temporal_roles | 4096 | 100 | 3 | 0.111 | 0.922 | 0.517 |
| flat_temporal_roles | 4096 | 200 | 3 | 0.000 | 0.663 | 0.332 |
| hybrid_chunked_plus_latest_cache | 1024 | 50 | 3 | 1.000 | 0.962 | 0.981 |
| hybrid_chunked_plus_latest_cache | 1024 | 100 | 3 | 1.000 | 0.908 | 0.954 |
| hybrid_chunked_plus_latest_cache | 1024 | 200 | 3 | 1.000 | 0.857 | 0.929 |
| hybrid_chunked_plus_latest_cache | 2048 | 50 | 3 | 1.000 | 1.000 | 1.000 |
| hybrid_chunked_plus_latest_cache | 2048 | 100 | 3 | 1.000 | 0.996 | 0.998 |
| hybrid_chunked_plus_latest_cache | 2048 | 200 | 3 | 1.000 | 0.993 | 0.997 |
| hybrid_chunked_plus_latest_cache | 4096 | 50 | 3 | 1.000 | 1.000 | 1.000 |
| hybrid_chunked_plus_latest_cache | 4096 | 100 | 3 | 1.000 | 1.000 | 1.000 |
| hybrid_chunked_plus_latest_cache | 4096 | 200 | 3 | 1.000 | 1.000 | 1.000 |
| overwrite_only | 1024 | 50 | 3 | 0.056 | 0.000 | 0.028 |
| overwrite_only | 1024 | 100 | 3 | 0.111 | 0.000 | 0.056 |
| overwrite_only | 1024 | 200 | 3 | 0.000 | 0.000 | 0.000 |
| overwrite_only | 2048 | 50 | 3 | 0.000 | 0.000 | 0.000 |
| overwrite_only | 2048 | 100 | 3 | 0.000 | 0.000 | 0.000 |
| overwrite_only | 2048 | 200 | 3 | 0.000 | 0.000 | 0.000 |
| overwrite_only | 4096 | 50 | 3 | 0.111 | 0.000 | 0.056 |
| overwrite_only | 4096 | 100 | 3 | 0.111 | 0.000 | 0.056 |
| overwrite_only | 4096 | 200 | 3 | 0.056 | 0.000 | 0.028 |

## Conclusion

- The best balanced strategy is `hybrid_chunked_plus_latest_cache` at `dim=2048`, `n_events=50`, with `balanced_score=1.000`.
- The hybrid latest-cache plus chunked-order strategy keeps latest-state and pairwise ordering measurable together, which avoids the earlier near-zero latest-state collapse.
- Overwrite-only storage still cleanly isolates the cost of dropping temporal-order structure.
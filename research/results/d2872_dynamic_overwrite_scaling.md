# D-2872 dynamic overwrite scaling

## Configuration

- `dims=[256, 1024, 2048]`
- `entity_counts=[20, 50]`
- `update_counts=[5, 20]`
- `properties=['location', 'action']`

## Summary

| dim | entities | updates | properties | condition | runs | mean_em | delta_vs_no_reset | location_em | action_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 20 | 5 | 2 | no_reset | 3 | 0.183 | +0.000 | 0.167 | 0.200 |
| 256 | 20 | 5 | 2 | perkey_reset | 3 | 0.242 | +0.058 | 0.283 | 0.200 |
| 256 | 20 | 20 | 2 | no_reset | 3 | 0.067 | +0.000 | 0.067 | 0.067 |
| 256 | 20 | 20 | 2 | perkey_reset | 3 | 0.150 | +0.083 | 0.100 | 0.200 |
| 256 | 50 | 5 | 2 | no_reset | 3 | 0.167 | +0.000 | 0.160 | 0.173 |
| 256 | 50 | 5 | 2 | perkey_reset | 3 | 0.270 | +0.103 | 0.253 | 0.287 |
| 256 | 50 | 20 | 2 | no_reset | 3 | 0.057 | +0.000 | 0.073 | 0.040 |
| 256 | 50 | 20 | 2 | perkey_reset | 3 | 0.140 | +0.083 | 0.120 | 0.160 |
| 1024 | 20 | 5 | 2 | no_reset | 3 | 0.167 | +0.000 | 0.167 | 0.167 |
| 1024 | 20 | 5 | 2 | perkey_reset | 3 | 0.267 | +0.100 | 0.233 | 0.300 |
| 1024 | 20 | 20 | 2 | no_reset | 3 | 0.042 | +0.000 | 0.033 | 0.050 |
| 1024 | 20 | 20 | 2 | perkey_reset | 3 | 0.108 | +0.067 | 0.117 | 0.100 |
| 1024 | 50 | 5 | 2 | no_reset | 3 | 0.150 | +0.000 | 0.153 | 0.147 |
| 1024 | 50 | 5 | 2 | perkey_reset | 3 | 0.263 | +0.113 | 0.260 | 0.267 |
| 1024 | 50 | 20 | 2 | no_reset | 3 | 0.037 | +0.000 | 0.013 | 0.060 |
| 1024 | 50 | 20 | 2 | perkey_reset | 3 | 0.100 | +0.063 | 0.107 | 0.093 |
| 2048 | 20 | 5 | 2 | no_reset | 3 | 0.208 | +0.000 | 0.267 | 0.150 |
| 2048 | 20 | 5 | 2 | perkey_reset | 3 | 0.225 | +0.017 | 0.233 | 0.217 |
| 2048 | 20 | 20 | 2 | no_reset | 3 | 0.058 | +0.000 | 0.067 | 0.050 |
| 2048 | 20 | 20 | 2 | perkey_reset | 3 | 0.125 | +0.067 | 0.133 | 0.117 |
| 2048 | 50 | 5 | 2 | no_reset | 3 | 0.193 | +0.000 | 0.233 | 0.153 |
| 2048 | 50 | 5 | 2 | perkey_reset | 3 | 0.247 | +0.053 | 0.267 | 0.227 |
| 2048 | 50 | 20 | 2 | no_reset | 3 | 0.040 | +0.000 | 0.053 | 0.027 |
| 2048 | 50 | 20 | 2 | perkey_reset | 3 | 0.117 | +0.077 | 0.133 | 0.100 |

## Conclusion

- `perkey_reset` is consistently better than `no_reset` across the current grid.
- The strongest observed gain is `+0.113` at `dim=1024`, `entities=50`, `updates=5`.
- The weakest observed gain is `+0.017` at `dim=2048`, `entities=20`, `updates=5`.
- The repo now mirrors keyed overwrite mechanics and direct reset-vs-no-reset comparison, but the absolute EM surface is still materially harsher than the lab-positive result, so this should still be treated as protocol alignment rather than numeric reproduction.
# Sequential Unbinding Scaling

## Configuration

- `dims=[256, 1024, 2048]`
- `hop_depths=[1, 2, 3]`
- `syntax_depths=[1, 2, 3]`
- `seeds=[0, 1, 2]`
- `n_chains=20`
- `n_sentences=25`

## Relation Chain Summary

| dim | depth | runs | mean_em |
| --- | --- | --- | --- |
| 256 | 1 | 3 | 0.467 |
| 256 | 2 | 3 | 0.183 |
| 256 | 3 | 3 | 0.083 |
| 1024 | 1 | 3 | 0.900 |
| 1024 | 2 | 3 | 0.867 |
| 1024 | 3 | 3 | 0.750 |
| 2048 | 1 | 3 | 1.000 |
| 2048 | 2 | 3 | 1.000 |
| 2048 | 3 | 3 | 1.000 |

## Hierarchical Syntax Summary

| dim | depth | runs | mean_em |
| --- | --- | --- | --- |
| 256 | 1 | 3 | 0.800 |
| 256 | 2 | 3 | 0.587 |
| 256 | 3 | 3 | 0.373 |
| 1024 | 1 | 3 | 1.000 |
| 1024 | 2 | 3 | 0.987 |
| 1024 | 3 | 3 | 0.907 |
| 2048 | 1 | 3 | 1.000 |
| 2048 | 2 | 3 | 0.987 |
| 2048 | 3 | 3 | 0.933 |

## Recommended Hop Bases

| dim | fitted_hop_base |
| --- | --- |
| 256 | 0.444 |
| 1024 | 0.913 |
| 2048 | 1.000 |

## Conclusion

- The relation-chain frontier now reports a fitted per-hop base instead of only broad dimension tiers.
- `d=1024` is the first strong runtime frontier, while `d>=2048` behaves as effectively exact for relation depth `1-3` in this repo-local mirror.
- The hierarchical side stays aligned with the same dimension sweep, which makes the hop-budget story easier to compare against other structured retrieval limits.
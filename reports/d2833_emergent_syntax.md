# D-2833 Emergent Syntax Reproduction

Generated on 2026-04-22.

## Question

Can HRR role-filler geometry make syntactically different forms of the same
meaning land near each other without training direct cross-pattern mappings?

## Protocol

- Encode 5 syntactic patterns: active, passive, relative, prepositional, and
  coordinated.
- Use shared semantic role bindings for subject, verb, and object.
- Add separate pattern and surface-variant bindings.
- Evaluate 5 synthetic domains x 30 triples x 3 seeds.
- Compare:
  - within-pattern paraphrases of the same triple,
  - cross-pattern forms of the same triple,
  - unrelated triples with unrelated syntax.

## Result

```text
seed  mean_within  mean_cross  random   cross_margin
0     0.888        0.645       0.003    0.642
1     0.888        0.653      -0.010    0.663
2     0.890        0.651      -0.005    0.656
```

## Interpretation

This local reproduction matches the D-2833 lab finding qualitatively and almost
numerically: same-meaning cross-pattern vectors remain strongly aligned, while
unrelated semantic/syntax pairs stay near orthogonal.

The important point is that no explicit active-to-passive, active-to-relative,
or other cross-pattern lookup was trained. The alignment is produced by shared
role-filler structure. This closes the previous roadmap gap that treated
emergent syntax as unimplemented.

## Limitation

This is controlled syntactic structure, not raw grammar induction from a corpus.
The next step is to feed extracted clauses and paraphrases from larger text into
the same measurement instead of synthetic triples.

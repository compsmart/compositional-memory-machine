# Typed Relation Fallback Real-Corpus Validation

This report validates the experimental typed relation fallback on curated
corpus-style facts and benchmark-style canonical queries.

## Summary

| fallback | category | pass_rate | exact_canonical_recovery_rate | typed_fallback_rate | cases |
| --- | --- | ---: | ---: | ---: | ---: |
| off | positive | 0.000 | 0.000 | 0.000 | 4 |
| off | negative | 1.000 | 0.000 | 0.000 | 2 |
| off | overall | 0.333 | 0.000 | 0.000 | 6 |
| on | positive | 0.250 | 0.250 | 0.250 | 4 |
| on | negative | 1.000 | 0.000 | 0.000 | 2 |
| on | overall | 0.500 | 0.167 | 0.167 | 6 |

## Case Results

| case_id | fallback | category | normalized | source | exact_canonical_recovery | passed |
| --- | --- | --- | --- | --- | ---: | ---: |
| worked_with_teamed_up_with | off | positive | teamed_up_with | self | 0 | 0 |
| published_notes_about_memo_on | off | positive | circulated_a_memo_on | self | 0 | 0 |
| proposed_as_pitched_as | off | positive | pitched_as | self | 0 | 0 |
| described_outlined | off | positive | outlined | self | 0 | 0 |
| negative_mentored | off | negative | mentored | self | 0 | 1 |
| negative_relocated_to | off | negative | relocated_to | self | 0 | 1 |
| worked_with_teamed_up_with | on | positive | worked_with | typed_fallback | 1 | 1 |
| published_notes_about_memo_on | on | positive | circulated_a_memo_on | self | 0 | 0 |
| proposed_as_pitched_as | on | positive | pitched_as | self | 0 | 0 |
| described_outlined | on | positive | outlined | self | 0 | 0 |
| negative_mentored | on | negative | mentored | self | 0 | 1 |
| negative_relocated_to | on | negative | relocated_to | self | 0 | 1 |

## Interpretation

- Positive corpus-style cases improved from `pass_rate=0.000`
  with fallback off to `pass_rate=0.250` with fallback on.
- Exact canonical recovery for positive cases improved from
  `exact_canonical_recovery_rate=0.000`
  to `exact_canonical_recovery_rate=0.250`.
- Negative safety cases with fallback on stayed at `pass_rate=1.000`,
  which is the key check against over-eager relation collapse.

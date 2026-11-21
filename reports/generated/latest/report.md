# Approved Standard-decision study

| system   |   n |   accuracy |   macro_f1 |   policy_f1 |   valid_inline_citation_coverage |   policy_exact_accuracy |
|:---------|----:|-----------:|-----------:|------------:|---------------------------------:|------------------------:|
| S0       |   3 |   0.333333 |       0.25 |           0 |                                0 |                       0 |
| S1       |   3 |   0.333333 |       0.25 |           0 |                                0 |                       0 |
| S2       |   3 |   0.333333 |       0.25 |           0 |                                0 |                       0 |

![Accuracy on three approved decisions](accuracy.svg)

FLAN-T5-small scored two possible outcomes and generated a rationale and policy for each input. Each fold excluded all three test cases and used only earlier dated documents, including for TF-IDF fitting. Three examples give coarse accuracy steps of 33 percentage points. The scores do not establish generalization. Inline citation coverage counts rationales containing a valid retrieved passage ID. It does not measure entailment. Policy exact accuracy requires matching the complete normalized label set.

## Retrieval sensitivity

| backend   |   chunk_size |   k | rerank   |   distinct_cases |   policy_hit |
|:----------|-------------:|----:|:---------|-----------------:|-------------:|
| hybrid    |          900 |   3 | False    |          2.33333 |            0 |
| hybrid    |          900 |   3 | True     |          3       |            0 |
| hybrid    |          900 |   6 | False    |          3.66667 |            0 |
| hybrid    |          900 |   6 | True     |          6       |            0 |
| lexical   |          480 |   3 | False    |          2       |            0 |
| lexical   |          480 |   3 | True     |          3       |            0 |
| lexical   |          480 |   6 | False    |          3.66667 |            0 |
| lexical   |          480 |   6 | True     |          6       |            0 |
| lexical   |          900 |   3 | False    |          2.33333 |            0 |
| lexical   |          900 |   3 | True     |          3       |            0 |
| lexical   |          900 |   6 | False    |          4       |            0 |
| lexical   |          900 |   6 | True     |          6       |            0 |

Distinct cases measures source diversity. Policy hit measures whether any retrieved source shares an annotated policy. Compare the two lexical sizes for chunk length, and the two backends at 900 for embedding effects. The reranker promotes the first passage from each distinct decision.

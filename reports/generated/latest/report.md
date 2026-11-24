# Approved Standard-decision study

| system   |   n |   accuracy |   macro_f1 |   policy_f1 |   valid_inline_citation_coverage |   policy_exact_accuracy |
|:---------|----:|-----------:|-----------:|------------:|---------------------------------:|------------------------:|
| S0       |  12 |   0.583333 |   0.555556 |   0.0555556 |                                0 |                       0 |
| S1       |  12 |   0.75     |   0.733333 |   0.0888889 |                                1 |                       0 |
| S2       |  12 |   0.5      |   0.485714 |   0.0888889 |                                1 |                       0 |

![Accuracy on 12 approved decisions](accuracy.svg)

Generator: `Qwen/Qwen2.5-0.5B-Instruct`. Dense retriever: `sentence-transformers/all-MiniLM-L6-v2`. The processed corpus contains 94 decisions and the human-approved review set contains 12 cases. The configured pretrained instruction model generated the outcome, rationale, policy labels and citation IDs. The retrieval corpus excluded all 12 test cases and used only documents published before the earliest test case, including for TF-IDF fitting. With 12 examples, accuracy changes in steps of 8.3 percentage points. The scores do not establish generalization. Inline citation coverage counts rationales containing a valid retrieved passage ID. It does not measure entailment. Policy exact accuracy requires matching the complete normalized label set.

## Retrieval sensitivity

| backend   |   chunk_size |   k | rerank   |   distinct_cases |   policy_hit |
|:----------|-------------:|----:|:---------|-----------------:|-------------:|
| hybrid    |          900 |   3 | False    |          2.5     |     0.333333 |
| hybrid    |          900 |   3 | True     |          3       |     0.416667 |
| hybrid    |          900 |   6 | False    |          4.08333 |     0.583333 |
| hybrid    |          900 |   6 | True     |          6       |     0.583333 |
| lexical   |          480 |   3 | False    |          2.16667 |     0.5      |
| lexical   |          480 |   3 | True     |          3       |     0.5      |
| lexical   |          480 |   6 | False    |          3.33333 |     0.5      |
| lexical   |          480 |   6 | True     |          6       |     0.666667 |
| lexical   |          900 |   3 | False    |          2.41667 |     0.416667 |
| lexical   |          900 |   3 | True     |          3       |     0.5      |
| lexical   |          900 |   6 | False    |          3.83333 |     0.583333 |
| lexical   |          900 |   6 | True     |          6       |     0.666667 |

Distinct cases measures source diversity. Policy hit measures whether any retrieved source shares an annotated policy. Compare the two lexical sizes for chunk length, and the two backends at 900 for embedding effects. The reranker promotes the first passage from each distinct decision.

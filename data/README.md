# Data inventory

| Path | Rows/items | Role |
|---|---:|---|
| `lexicon/turkish_framenet_2_annotated.csv` | 8,549 | released labelled lexical resource |
| `lexicon/train.csv` | 6,876 | fixed training split |
| `lexicon/test.csv` | 1,673 | fixed held-out test split |
| `lexicon/expansion_target_unlabelled.csv` | 9,238 records | unlabelled expansion pool; 9,237 distinct synset IDs |
| `expansion_eval/expert_annotations.csv` | 1,000 sampled / 967 completed | candidate-space expert evaluation |
| `expansion_eval/model_predictions.csv` | 1,000 | archived system predictions |
| `resource_quality/audit_records.csv` | 400 | anonymised blind reannotation and independent expert-adjudication records |
| `resource_quality/blind_reannotation_expert_adjudication_400.xlsx` | 400 | released audit workbook with summary, codebook, notes, and audit records |
| `frame_inventory/framenet_1_7_frames.json` | 1,221 frames | diagnostic FrameNet-definition retrieval |

The annotated lexical resource contains 8,549 distinct synset IDs. The expansion pool contains one exact duplicate record, giving 9,238 records and 9,237 distinct IDs. The annotated and expansion ID sets are disjoint; original records are preserved for reproducibility.

Missing definitions in the fixed lexical split are preserved rather than silently imputed: four test definitions and ten training definitions are empty.

The audit reference labels and the original benchmark labels are tracked separately. The [sensitivity supplement](../sensitivity/README.md) provides their ID and label crosswalk and the 101 expert-accepted reannotation decisions. Its separate scoring scenario applies only the 15 accepted test decisions, changing 14 frozen test labels; training labels and the original benchmark files remain unchanged. See the [audit protocol](resource_quality/PROTOCOL.md) for the blind reannotation and expert-adjudication procedure.

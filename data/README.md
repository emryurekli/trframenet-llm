# Data inventory

| Path | Rows/items | Role |
|---|---:|---|
| `lexicon/turkish_framenet_2_annotated.csv` | 8,549 | released labelled lexical resource |
| `lexicon/train.csv` | 6,876 | fixed training split |
| `lexicon/test.csv` | 1,673 | fixed held-out test split |
| `lexicon/expansion_target_unlabelled.csv` | 9,238 | unlabelled expansion pool |
| `expansion_eval/expert_annotations.csv` | 1,000 sampled / 967 completed | candidate-space expert evaluation |
| `expansion_eval/model_predictions.csv` | 1,000 | archived system predictions |
| `resource_quality/audit_records.csv` | 400 | anonymised blind reannotation and independent expert-adjudication records |
| `resource_quality/blind_reannotation_expert_adjudication_400.xlsx` | 400 | submission-ready workbook with summary, codebook, notes, and audit records |
| `frame_inventory/framenet_1_7_frames.json` | 1,221 frames | diagnostic FrameNet-definition retrieval |

Missing definitions in the fixed lexical split are preserved rather than silently imputed: four test definitions and ten training definitions are empty.

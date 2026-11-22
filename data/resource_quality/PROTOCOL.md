# Blind reannotation and independent expert adjudication

The resource-quality study contains 400 released Turkish synset-definition entries, balanced across four diagnostic strata:

- 100 rare-frame items;
- 100 frequent-frame items;
- 100 repeated or polysemous surface forms;
- 100 multiword expressions.

## Stage 1: blind reannotation

Annotator A1 received the synset, Turkish definition, canonical FrameNet inventory, and written guidelines. A1 did not receive the released label or any model output. A1 selected one primary frame, optional alternatives, or `NO_SUITABLE_FRAME`, and recorded confidence and difficulty notes.

## Stage 2: independent expert adjudication

A separate expert, A2, reviewed each record using only the synset, Turkish definition, A1 decision, released label, and FrameNet evidence. A2 did not inspect model predictions. A2 assigned one of four outcomes:

1. `Direct match — released label supported`
2. `Released label retained`
3. `Both plausible`
4. `New label accepted`

The audit contains 81 direct matches, 179 retained released labels, 39 both-plausible cases, and 101 accepted reannotations. The released label is retained or judged plausible in 299/400 items. Because the four strata are deliberately balanced, these rates are diagnostic and are not corpus-wide error-prevalence estimates. Five omitted A1 decisions were restored as direct matches after confirmation from the annotation record.

This is a blind reannotation followed by independent expert adjudication. It is not a parallel two-blind-annotator IAA study.

## Benchmark and correction release

The audit reference label is the released label recorded in the annotation workbook; it is tracked separately from the original frozen benchmark label. The main benchmark results use the original train/test labels. The [sensitivity supplement](../../sensitivity/README.md) provides the audit-to-benchmark crosswalk and the 101 expert-accepted reannotation decisions as a separate record.

Of the 400 audited entries, 330 are training entries and 70 are test entries. The accepted decisions concern 86 training entries and 15 test entries. Sensitivity analysis applies only the 15 accepted test decisions to a separate scoring scenario; one accepted label already matches the original benchmark label, so 14 test labels change. Training labels, model predictions, and all other benchmark labels remain fixed. These audit decisions do not imply that the original benchmark files were replaced.

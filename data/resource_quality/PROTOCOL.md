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

Reported model results use the original frozen train/test labels to avoid post-hoc benchmark alteration. The 101 expert-accepted corrections are provided in the audit records as a versioned adjudication record, while the original labels remain available for exact reproduction of the paper results.

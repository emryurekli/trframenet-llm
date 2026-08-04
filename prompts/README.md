# Prompt templates

- `frame_prediction_few_shot.txt`: primary synset–definition frame selection with five retrieved, frame-diverse examples.
- `frame_prediction_zero_shot.txt`: the same task without demonstrations.
- `judge.txt`: the preliminary 1–5 appropriateness-scoring prompt reported only in the appendix.

The raw local/API outputs used in the primary benchmark are archived under `results/raw_outputs/`. Candidate-conditioned LoRA is prompt conditioning, not vocabulary-constrained decoding.

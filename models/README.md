# Model artefacts

`lora_adapter_config.json` records the exact PEFT/LoRA configuration used for Qwen2.5-7B-Instruct. The adapter weights are not included in this repository. The training script, fixed training split, prompt format, archived predictions, evaluation code, and configuration support retraining and evaluation. The primary benchmark and label-sensitivity results can be reproduced from the archived predictions without the adapter weights.

The frozen multilingual-encoder classifier and label encoder are provided under `artifacts/encoder/`. Large train/test embedding matrices were omitted because they are deterministic caches that can be regenerated from the released split and checkpoint.

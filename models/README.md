# Model artefacts

`lora_adapter_config.json` records the exact PEFT/LoRA configuration used for Qwen2.5-7B-Instruct. The adapter weights are not included in the anonymous review package. The fixed training split, prompt format, archived predictions, evaluation code, and configuration are provided so that the adapter can be regenerated or checked after release.

The frozen multilingual-encoder classifier and label encoder are provided under `artifacts/encoder/`. Large train/test embedding matrices were omitted because they are deterministic caches that can be regenerated from the released split and checkpoint.

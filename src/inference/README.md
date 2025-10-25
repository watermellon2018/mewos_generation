# LoRA Generation Script

Flexible Python script for generating images with your trained LoRA model.

## Usage Examples

### Single Image Generation
```bash
python src/inference/generation.py \
    --base_model weights/model_768.safetensors \
    --lora_path output/lora/last.safetensors \
    --prompt "masterpiece, best quality, mewos cat in forest" \
    --output_dir ./generated_images
```

### Batch Generation from Prompts File
<!-- ```bash
python src/inference/generation.py \
    --base_model weights/model_768.safetensors \
    --lora_path output/lora/last.safetensors \
    --prompts_file src/inference/example_prompts.txt \
    --n_samples 3 \
    --output_dir ./generated_images
``` -->

### Generation from Config File
```bash
python src/inference/generation.py \
    --base_model weights/model_768.safetensors \
    --lora_path output/lora/last.safetensors \
    --config src/inference/example_config.json \
    --output_dir ./generated_images
```

## Parameters

- `--base_model`: Path to base Stable Diffusion model
- `--lora_path`: Path to your trained LoRA weights
- `--output_dir`: Directory to save generated images
- `--prompt`: Single prompt for generation
- `--prompts_file`: Text file with prompts (one per line)
- `--config`: JSON config file for advanced batch generation
- `--width/--height`: Image dimensions (default: 768x768)
- `--steps`: Sampling steps (default: 28)
- `--cfg_scale`: CFG scale (default: 7.5)
- `--seed`: Random seed (auto-generated if not specified)
- `--n_samples`: Number of samples per prompt
- `--negative_prompt`: Negative prompt override

## Config File Format

```json
{
    "prompts": ["prompt1", "prompt2"],
    "n_samples": 2,
    "width": 768,
    "height": 768,
    "steps": 28,
    "cfg_scale": 7.5,
    "negative_prompt": "low quality, blurry",
    "seeds": [12345, 67890]
}
```

## Features

- ✅ Single and batch generation
- ✅ Multiple input formats (prompt, file, config)
- ✅ Flexible parameter control
- ✅ Automatic seed generation
- ✅ Error handling and validation
- ✅ Progress tracking
- ✅ Configurable output naming

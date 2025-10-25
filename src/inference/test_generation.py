"""
Генерация без использования Diffusers pipeline
Прямой inference через kohya модули
Обходит проблемы с несовместимостью diffusers 0.35+
"""

import torch
import sys
import os
from PIL import Image
import numpy as np

# Добавляем путь к kohya
sys.path.insert(0, "libs/sd-scripts")

from safetensors.torch import load_file
import library.model_util as model_util
from networks.lora import LoRANetwork
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
import torch.nn.functional as F

# ============================================================================
# Настройки
# ============================================================================

MODEL_PATH = "weights/model_768.safetensors"
LORA_PATH = "output/lora/last.safetensors"
LORA_WEIGHT = 1.0
OUTPUT_DIR = "generated_images"
DEVICE = "cuda"

# Параметры генерации
PROMPT = "mewos sitting, beautiful lighting, detailed fur"
NEGATIVE_PROMPT = "low quality, blurry, bad anatomy"
HEIGHT = 768
WIDTH = 768
NUM_STEPS = 30
GUIDANCE_SCALE = 7.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("Генерация через нативный inference (без Diffusers pipeline)")
print("="*70)

# ============================================================================
# Загрузка моделей
# ============================================================================

print("\n1. Загрузка базовой модели...")
text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(
    v2=True, ckpt_path=MODEL_PATH
)

text_encoder = text_encoder.to(DEVICE, dtype=torch.float16)
vae = vae.to(DEVICE, dtype=torch.float16)
unet = unet.to(DEVICE, dtype=torch.float16)

print("✓ Модели загружены")

# ============================================================================
# Загрузка LoRA
# ============================================================================

print("\n2. Загрузка LoRA...")
lora_sd = load_file(LORA_PATH)

modules_dim = {}
modules_alpha = {}

for key, value in lora_sd.items():
    if "." not in key:
        continue
    lora_name = key.split(".")[0]
    if "alpha" in key:
        modules_alpha[lora_name] = value
    elif "lora_down" in key:
        modules_dim[lora_name] = value.size()[0]

for key in modules_dim.keys():
    if key not in modules_alpha:
        modules_alpha[key] = modules_dim[key]

network = LoRANetwork(
    text_encoder=text_encoder,
    unet=unet,
    multiplier=LORA_WEIGHT,
    modules_dim=modules_dim,
    modules_alpha=modules_alpha,
)

network.load_state_dict(lora_sd, strict=False)
network.apply_to(text_encoder, unet)
network.to(DEVICE, dtype=torch.float16)

print(f"✓ LoRA загружена (модулей: {len(modules_dim)})")

# ============================================================================
# Загрузка tokenizer и scheduler
# ============================================================================

print("\n3. Загрузка tokenizer и scheduler...")
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2", subfolder="scheduler")

print("✓ Tokenizer и scheduler загружены")

# ============================================================================
# Подготовка промптов
# ============================================================================

print("\n4. Подготовка промптов...")

# Токенизация
text_input = tokenizer(
    PROMPT,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)

uncond_input = tokenizer(
    NEGATIVE_PROMPT,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)

# Получаем embeddings
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(DEVICE))[0]

# Объединяем для classifier-free guidance
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

print(f"✓ Embeddings готовы: {text_embeddings.shape}")

# ============================================================================
# Генерация
# ============================================================================

print("\n5. Генерация изображения...")
print(f"   Промпт: {PROMPT}")
print(f"   Разрешение: {WIDTH}x{HEIGHT}")
print(f"   Шагов: {NUM_STEPS}")
print(f"   Guidance scale: {GUIDANCE_SCALE}")
print(f"   Seed: {SEED}")

# Устанавливаем seed
generator = torch.Generator(device=DEVICE).manual_seed(SEED)

# Инициализируем latents
# For SD 2.0, in_channels is always 4
in_channels = 4
latents = torch.randn(
    (1, in_channels, HEIGHT // 8, WIDTH // 8),
    generator=generator,
    device=DEVICE,
    dtype=torch.float16
)

# Настраиваем scheduler
scheduler.set_timesteps(NUM_STEPS)
latents = latents * scheduler.init_noise_sigma

# Деnoising loop
print("\n   Прогресс:")
for i, t in enumerate(scheduler.timesteps):
    # Дублируем latents для CFG
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # Предсказание шума
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
    
    # Classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
    
    # Обновляем latents
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Прогресс
    if (i + 1) % 5 == 0 or i == 0 or i == len(scheduler.timesteps) - 1:
        print(f"   Шаг {i+1}/{NUM_STEPS}")

print("\n6. Декодирование изображения...")

# Декодируем latents в изображение
with torch.no_grad():
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample

# Постобработка
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
image = (image[0] * 255).round().astype("uint8")
image = Image.fromarray(image)

# Сохранение
output_path = os.path.join(OUTPUT_DIR, "mewos_native.png")
image.save(output_path, quality=95)

print(f"\n✓ Изображение сохранено: {output_path}")

print("\n" + "="*70)
print("✓ Генерация завершена!")
print("="*70)
print("\nЭтот метод обходит проблемы с Diffusers pipeline")
print("и работает напрямую с моделями через kohya.")
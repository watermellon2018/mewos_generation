# Makefile для тестирования разных чекпоинтов LoRA

# === Параметры ===
CHECKPOINT := weights/768-v-ema.safetensors
LORA_DIR := output/lora
OUTPUT_BASE := checkpoint_tests
PROMPT := meows cat sitting, looking at camera, detailed fur, front view

# Находим все чекпоинты
CHECKPOINTS := $(shell find $(LORA_DIR) -name "epoch-*.safetensors" | sort)

# === Основные цели ===
.PHONY: all test clean

all: test

test:
	@echo "=================================================="
	@echo "Тестирование всех чекпоинтов LoRA"
	@echo "=================================================="
	@mkdir -p "$(OUTPUT_BASE)"

	@# Проверяем, есть ли файлы
	@if ! find "$(LORA_DIR)" -name "epoch-*.safetensors" -print -quit | grep -q .; then \
		echo "Чекпоинты не найдены в $(LORA_DIR)"; \
		exit 1; \
	fi

	@echo "Найдено чекпоинтов: $$(find "$(LORA_DIR)" -name 'epoch-*.safetensors' | wc -l)"
	@echo ""

	@find "$(LORA_DIR)" -name "epoch-*.safetensors" -print | sort | while IFS= read -r LORA_WEIGHTS; do \
		[ -z "$$LORA_WEIGHTS" ] && continue; \
		CHECKPOINT_NAME=$$(basename "$$LORA_WEIGHTS" .safetensors); \
		OUTPUT_DIR="$(OUTPUT_BASE)/$$CHECKPOINT_NAME"; \
		echo "=================================================="; \
		echo "Тестирование: $$CHECKPOINT_NAME"; \
		echo "=================================================="; \
		mkdir -p "$$OUTPUT_DIR"; \
		python libs/sd-scripts/gen_img.py \
			--ckpt="$(CHECKPOINT)" \
			--v2 \
			--v_parameterization \
			--network_weights="$$LORA_WEIGHTS" \
			--network_mul=1.0 \
			--prompt="$(PROMPT)" \
			--outdir="$$OUTPUT_DIR" \
			--images_per_prompt=1 \
			--steps=30 \
			--fp16 \
			--xformers || exit 1; \
		echo "✓ $$CHECKPOINT_NAME завершен"; \
		echo ""; \
	done || exit 1

	@echo "=================================================="
	@echo "Все тесты завершены!"
	@echo "  Результаты в: $(OUTPUT_BASE)"
	@echo "=================================================="
	@echo "Теперь сравните изображения и выберите лучший чекпоинт (обычно 20–40 эпох)."

clean:
	@echo "🧹 Очистка результатов..."
	@rm -rf "$(OUTPUT_BASE)"
	@echo "Готово."

SHELL := /bin/bash
ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(ROOT_DIR)data/lora/big
RESIZED_IMGS_DIR := "$(ROOT_DIR)data/lora/resized/20_cat mewos"
MAX_SIZE := 768
LIBS_DIR := $(ROOT_DIR)libs

# Параметры для обучения
# https://huggingface.co/stabilityai/stable-diffusion-2
PRETRAINED_MODEL:= $(ROOT_DIR)weights/768-v-ema.safetensors
OUTPUT_DIR:= $(ROOT_DIR)output/lora
LOGGING_DIR:= $(ROOT_DIR)logs
NETWORK_ALPHA:= 128
NETWORK_DIM:= 128
RESOLUTION:= 768,768
TRAIN_BATCH_SIZE:= 8
MAX_TRAIN_EPOCHS:= 100
LEARNING_RATE:= 1e-4
PRIOR_LOSS_WEIGHT:= 0

# ============================================================================
# Флаги для SD 2.0 и оптимизации VRAM
# ============================================================================
# Флаги для Stable Diffusion 2.0 с v-parameterization
V2_FLAG := --v2
V_PARAM_FLAG := --v_parameterization

# Оптимизация VRAM
OPTIMIZER_TYPE := AdamW8bit
XFORMERS := --xformers
GRADIENT_CHECKPOINTING := --gradient_checkpointing
CACHE_LATENTS := --cache_latents

# Создание недостающие директории
prepare_dirs:
	@echo "Create dir for saving of resized images..."
	mkdir -p $(RESIZED_IMGS_DIR)
	@echo "Done!"

# python $(LIBS_DIR)/sd-scripts/tools/resize_images_to_resolution.py $(DATA_DIR) $(RESIZED_IMGS_DIR) \
	 	# --max_resolution $(MAX_SIZE)x$(MAX_SIZE)
resize:
	@echo "Dataset: $(DATA_DIR)"
	@echo "Resizing images to max size $(MAX_SIZE)..."
	python src/utils/resize.py --src $(DATA_DIR) --dst $(RESIZED_IMGS_DIR) --size $(MAX_SIZE)
	cp $(DATA_DIR)/*.txt $(RESIZED_IMGS_DIR)/
	@echo "Done!"


train:
	@echo "Запускаем обучение модели..."
	accelerate launch $(LIBS_DIR)/sd-scripts/train_network.py \
		--pretrained_model_name_or_path="$(PRETRAINED_MODEL)" \
		--train_data_dir="$(ROOT_DIR)data/lora/resized" \
		--output_dir="$(OUTPUT_DIR)" \
		--logging_dir="$(LOGGING_DIR)" \
		--network_module="networks.lora" \
		--network_alpha="$(NETWORK_ALPHA)" \
		--network_dim="$(NETWORK_DIM)" \
		--resolution=$(RESOLUTION) \
		--train_batch_size=$(TRAIN_BATCH_SIZE) \
		--max_train_epochs=$(MAX_TRAIN_EPOCHS) \
		--learning_rate=$(LEARNING_RATE) \
		--lr_scheduler="constant" \
		--mixed_precision="fp16" \
		--save_every_n_epochs=5 \
		--caption_extension=".txt" \
		--prior_loss_weight=$(PRIOR_LOSS_WEIGHT) \
		--save_model_as="safetensors" \
		--network_dropout=0.2 \
		$(V2_FLAG) \
		$(V_PARAM_FLAG) \
		$(XFORMERS) \
		$(GRADIENT_CHECKPOINTING) \
		$(CACHE_LATENTS)

	@echo "Обучение модели завершено!"


all: prepare_dirs resize train
# тут есть регулязицонные картинки, их отключили prior_loss_weight=0, они нужны для предотвращения переобучения
# мы просто добавляем другие объекты. Потом можно добаввить.
# all:  train 

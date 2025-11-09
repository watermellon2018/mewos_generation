import torch
import argparse
from safetensors.torch import load_file, save_file

def convert_lora_4d_to_2d(lora_path, output_path):
    print(f"Загрузка LoRA из: {lora_path}")
    weights = load_file(lora_path)
    converted_weights = {}
    converted_count = 0

    for key, value in weights.items():
        if len(value.shape) == 4 and value.shape[2] == 1 and value.shape[3] == 1:
            converted_weights[key] = value.squeeze()
            print(f"  Конвертация {key}: {value.shape} -> {converted_weights[key].shape}")
            converted_count += 1
        else:
            converted_weights[key] = value

    if converted_count > 0:
        print(f"\nКонвертировано {converted_count} тензоров.")
        print(f"Сохранение в: {output_path}")
        save_file(converted_weights, output_path)
        print("✓ Готово!")
    else:
        print("Не найдено 4D тензоров для конвертации.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True, help="Путь к исходному LoRA файлу (.safetensors)")
    parser.add_argument("--output_path", type=str, required=True, help="Путь для сохранения конвертированного LoRA файла (.safetensors)")
    args = parser.parse_args()

    convert_lora_4d_to_2d(args.lora_path, args.output_path)


"""
Утилита для конвертации LoRA весов из формата Conv2d (4D) в формат Linear (2D)
Это нужно когда модель обучена с --enable_bucket в kohya
"""

import argparse
from safetensors.torch import load_file, save_file
import os

def convert_lora_weights(input_path, output_path=None, verbose=True):
    """
    Конвертирует LoRA веса из 4D (Conv2d) в 2D (Linear) формат
    
    Args:
        input_path: Путь к исходному .safetensors файлу
        output_path: Путь для сохранения (если None, добавляется _converted)
        verbose: Выводить детальную информацию
    """
    
    if not os.path.exists(input_path):
        print(f"✗ Ошибка: Файл не найден: {input_path}")
        return False
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted{ext}"
    
    print(f"Загрузка весов из: {input_path}")
    state_dict = load_file(input_path)
    
    print(f"Всего параметров: {len(state_dict)}")
    
    # Анализ весов
    conv2d_count = 0
    linear_count = 0
    other_count = 0
    
    for key, value in state_dict.items():
        if len(value.shape) == 4:
            conv2d_count += 1
        elif len(value.shape) == 2:
            linear_count += 1
        else:
            other_count += 1
    
    print(f"\nАнализ весов:")
    print(f"  Conv2d (4D): {conv2d_count}")
    print(f"  Linear (2D): {linear_count}")
    print(f"  Другие: {other_count}")
    
    if conv2d_count == 0:
        print("\n✓ Веса уже в правильном формате (2D), конвертация не требуется")
        return True
    
    # Конвертация
    print(f"\nКонвертация {conv2d_count} весов из 4D в 2D...")
    new_state_dict = {}
    converted_count = 0
    
    for key, value in state_dict.items():
        if len(value.shape) == 4:
            # Проверяем, что последние два измерения равны 1
            if value.shape[-2:] == (1, 1):
                # Убираем последние два измерения
                new_value = value.squeeze(-1).squeeze(-1)
                new_state_dict[key] = new_value
                converted_count += 1
                
                if verbose:
                    print(f"  ✓ {key}")
                    print(f"    {value.shape} -> {new_value.shape}")
            else:
                print(f"  ⚠ Пропущен {key}: неожиданная форма {value.shape}")
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value
    
    print(f"\n✓ Конвертировано весов: {converted_count}")
    
    # Сохранение
    print(f"\nСохранение в: {output_path}")
    save_file(new_state_dict, output_path)
    
    # Проверка размера файла
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\nРазмер файла:")
    print(f"  Исходный: {input_size:.2f} MB")
    print(f"  Конвертированный: {output_size:.2f} MB")
    
    print(f"\n{'='*70}")
    print("✓ Конвертация успешно завершена!")
    print(f"{'='*70}")
    print(f"\nТеперь используйте файл: {output_path}")
    print("для генерации изображений с помощью Diffusers")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Конвертация LoRA весов из Conv2d (4D) в Linear (2D) формат",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовая конвертация (создаст output/lora/last_converted.safetensors)
  python convert_lora_weights.py output/lora/last.safetensors

  # Указать выходной файл
  python convert_lora_weights.py output/lora/last.safetensors -o output/lora/last_fixed.safetensors

  # Тихий режим (без детальной информации)
  python convert_lora_weights.py output/lora/last.safetensors --quiet

Эта утилита нужна когда LoRA обучена с флагом --enable_bucket в kohya-ss/sd-scripts,
что создает веса в формате Conv2d (4D тензоры), несовместимые со стандартным API Diffusers.
        """
    )
    
    parser.add_argument(
        "input",
        help="Путь к исходному .safetensors файлу LoRA"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения конвертированного файла (по умолчанию: input_converted.safetensors)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Не выводить детальную информацию о каждом конвертированном весе"
    )
    
    args = parser.parse_args()
    
    success = convert_lora_weights(
        args.input,
        args.output,
        verbose=not args.quiet
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())

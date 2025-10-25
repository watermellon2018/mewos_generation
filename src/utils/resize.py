import argparse
import os

from PIL import Image


def resize_images_preserve_aspect(src_dir, dst_dir, target_size, quality=95):
    os.makedirs(dst_dir, exist_ok=True)
    print(f"Dataset: {src_dir}")
    print(f"Изменение размера изображений (max side -> {target_size}px), без кропа...")
    print("Сохраняем пропорции: большая сторона станет равна target_size.\n")

    supported = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
    files = sorted(os.listdir(src_dir))
    for filename in files:
        if not filename.lower().endswith(supported):
            continue

        src_path = os.path.join(src_dir, filename)
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                w, h = img.size

                if w == 0 or h == 0:
                    print(f"Пропущено (некорректный размер): {filename}")
                    continue

                # если обе стороны уже <= target_size, можно либо оставить, либо upscale — оставим без изменений
                if max(w, h) <= target_size:
                    out_img = img.copy()
                    out_size = (w, h)
                else:
                    if w >= h:
                        new_w = target_size
                        new_h = int(h * (target_size / w))
                    else:
                        new_h = target_size
                        new_w = int(w * (target_size / h))

                    out_img = img.resize((new_w, new_h), Image.LANCZOS)
                    out_size = (new_w, new_h)

                dst_path = os.path.join(dst_dir, filename)
                out_img.save(dst_path, quality=quality)
                print(f"Обработан: {filename} -> {out_size[0]}x{out_size[1]}")
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")

    print("\nГотово!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images preserving aspect ratio (no crop).")
    parser.add_argument("--src", required=True, help="Путь к исходным изображениям")
    parser.add_argument("--dst", required=True, help="Путь для сохранения обработанных изображений")
    parser.add_argument("--size", type=int, default=768, help="Максимальный размер стороны (в пикселях)")
    parser.add_argument("--quality", type=int, default=95, help="Качество при сохранении (1-100)")
    args = parser.parse_args()

    resize_images_preserve_aspect(args.src, args.dst, args.size, args.quality)

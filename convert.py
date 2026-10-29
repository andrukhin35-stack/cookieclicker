"""
convert.py — подготовка датасета для YOLO (Задача 2, пункты 5-10).

Что делает:
  1. Берёт скриншоты (.png) и разметку X-AnyLabeling (.json) из папки SOURCE_DIR.
  2. Переводит каждый прямоугольник в формат YOLO HBB:
         <номер_класса> <x_центра> <y_центра> <ширина> <высота>
     (все числа от 0 до 1 — доли от размера картинки).
  3. Раскладывает данные по папкам:
         images/train, images/val   — картинки (80% / 20%)
         labels/train, labels/val   — txt-файлы с разметкой

  4. Создаёт data.yaml по списку классов из class.txt.

Список классов берётся из class.txt (по одному названию на строку,
порядок важен: первая строка = класс 0, вторая = класс 1 и т.д.).

Запуск:
    python convert.py                 # берёт .png + .json из текущей папки
    python convert.py . screenshots   # из нескольких папок сразу
"""

import json
import os
import random
import shutil
import sys

SOURCE_DIRS = ["."]       # где лежат исходные .png + .json
CLASS_FILE = "class.txt"  # список классов
VAL_PART = 0.2            # 20% картинок уходит в val
SEED = 42                 # чтобы разбиение было одинаковым при каждом запуске


def read_classes(path):
    with open(path, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise SystemExit(f"Файл {path} пустой — впиши туда хотя бы 'game'")
    return classes


def json_to_yolo(json_path, classes):
    """Переводит один .json от X-AnyLabeling в строки формата YOLO."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    w, h = data["imageWidth"], data["imageHeight"]
    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in classes:
            print(f"  ! неизвестный класс '{label}' в {json_path} — пропускаю")
            continue

        xs = [p[0] for p in shape["points"]]
        ys = [p[1] for p in shape["points"]]
        # обрезаем рамку по краям картинки (если обвели чуть за край)
        x1, x2 = max(0, min(xs)), min(w, max(xs))
        y1, y2 = max(0, min(ys)), min(h, max(ys))

        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{classes.index(label)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def safe_name(path, used):
    """Имя файла без пробелов и русских букв, чтобы не было проблем с путями."""
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    name = "img_" + (digits or str(len(used)))
    while name in used:          # если вдруг имена совпали
        name += "_"
    used.add(name)
    return name


def write_data_yaml(classes):
    names = ", ".join(f"'{c}'" for c in classes)
    with open("data.yaml", "w", encoding="utf-8") as f:
        f.write("path: ./\n"
                "train: images/train\n"
                "val: images/val\n"
                "\n"
                f"nc: {len(classes)}  # количество классов\n"
                f"names: [{names}]  # названия классов\n")


def main(source_dirs=SOURCE_DIRS):
    classes = read_classes(CLASS_FILE)
    print("Классы:", classes)

    # пары картинка + разметка
    pairs = []
    for src in source_dirs:
        for name in sorted(os.listdir(src)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                json_path = os.path.join(src, os.path.splitext(name)[0] + ".json")
                if os.path.exists(json_path):
                    pairs.append((os.path.join(src, name), json_path))
                else:
                    print(f"  ! нет разметки для {name} — пропускаю")

    if not pairs:
        raise SystemExit("Не нашёл ни одной пары картинка + .json")

    random.seed(SEED)
    random.shuffle(pairs)
    n_val = max(1, round(len(pairs) * VAL_PART))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    # чистим старый датасет, чтобы одна картинка не попала и в train, и в val
    for folder in ("images", "labels"):
        shutil.rmtree(folder, ignore_errors=True)

    used = set()
    for split, items in splits.items():
        os.makedirs(f"images/{split}", exist_ok=True)
        os.makedirs(f"labels/{split}", exist_ok=True)
        for img, js in items:
            stem = safe_name(img, used)
            ext = os.path.splitext(img)[1].lower()
            shutil.copy(img, f"images/{split}/{stem}{ext}")
            with open(f"labels/{split}/{stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(json_to_yolo(js, classes)) + "\n")

    write_data_yaml(classes)
    print(f"Готово: train = {len(splits['train'])}, val = {len(splits['val'])}")
    print("data.yaml обновлён")


if __name__ == "__main__":
    # переходим в папку проекта, чтобы скрипт работал откуда угодно
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1:] or SOURCE_DIRS)

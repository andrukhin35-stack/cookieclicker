"""
auto_label.py — автоматическая разметка данных (пункт 2 отчёта).

Идея: вручную размечаем только первые ~30 картинок и обучаем модель.
Дальше уже обученная модель сама размечает новые скриншоты,
а человек только проверяет и поправляет ошибки.

Что делает скрипт:
  - берёт все картинки из папки screenshots (её заполняет screener.py);
  - прогоняет их через модель best.pt;
  - для классов, которых модель ещё не знает, использует простые правила:
      upgrade  — у здания, которое можно купить, цена написана ЗЕЛЁНЫМ
                 (у недоступного — красным);
      close_ad — кнопка закрытия рекламы: серый кружок с белым крестиком;
    правила рассчитаны на игру, открытую на весь экран 1920x1080;
  - рядом с каждой картинкой создаёт .json в формате X-AnyLabeling.

После этого:
  1. открой папку screenshots в X-AnyLabeling — рамки уже будут нарисованы;
  2. поправь неточные рамки и дорисуй то, что правила не нашли (например, bonus);
  3. запусти  python convert.py . screenshots  — и  python train.py

Запуск:  python auto_label.py            (папка screenshots)
         python auto_label.py моя_папка  (любая другая папка)
"""

import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "runs/detect/cookie_detector/weights/best.pt"
CLASS_FILE = "class.txt"
CONFIDENCE = 0.5      # рамки с уверенностью ниже этой не сохраняем
OVERWRITE = False     # True — перезаписывать уже готовые .json (осторожно: затрёт ручные правки)


def find_upgrades(img):
    """Ищет в магазине здания с ЗЕЛЁНОЙ ценой (их можно купить).
    Возвращает список рамок [x1, y1, x2, y2]."""
    pixels = np.asarray(img.convert("RGB")).astype(int)
    height, width, _ = pixels.shape
    k = width / 1920                           # координаты подобраны для 1920x1080
    column = pixels[:, int(1290 * k):int(1470 * k)]   # полоска, где пишутся цены
    r, g, b = column[..., 0], column[..., 1], column[..., 2]
    green = (g > 170) & (r < 140) & (b < 140)
    green_per_row = green.sum(axis=1)

    # склеиваем соседние строки пикселей в "строчки текста"
    boxes, start = [], None
    for y in range(height + 1):
        has_green = y < height and green_per_row[y] > 2
        if has_green and start is None:
            start = y
        elif not has_green and start is not None:
            # маленький разрыв внутри цифр не считаем концом строки
            if y < height and green_per_row[y:y + 4].sum() > 0:
                continue
            text_h = y - start
            if 6 * k <= text_h <= 18 * k and green_per_row[start:y].sum() >= 25 * k:
                center = (start + y) / 2
                # рамка на всю строку магазина: от иконки здания до числа справа
                boxes.append([1225 * k, center - 46 * k, 1555 * k, center + 24 * k])
            start = None
    return boxes


def find_close_buttons(img):
    """Ищет кнопку закрытия рекламы: серый круг (~53x53 px) с белым крестиком.
    Возвращает список рамок [x1, y1, x2, y2]."""
    pixels = np.asarray(img.convert("RGB")).astype(int)
    k = pixels.shape[1] / 1920
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    gray = (abs(r - g) < 12) & (abs(g - b) < 16) & (r > 80) & (r < 160)
    white = pixels.min(axis=2) > 200

    # ищем отдельные серые пятна подходящего размера и почти квадратной формы
    count, _, stats, _ = cv2.connectedComponentsWithStats(gray.astype(np.uint8), 8)
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        if 45 * k <= w <= 65 * k and abs(w - h) <= 6 * k and area >= 0.4 * w * h:
            if white[y:y + h, x:x + w].mean() > 0.05:     # внутри есть белый крестик
                boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


RULES = {"upgrade": find_upgrades, "close_ad": find_close_buttons}


def make_shape(label, x1, y1, x2, y2, score):
    """Прямоугольник в формате X-AnyLabeling / LabelMe."""
    return {
        "label": label,
        "score": round(score, 3),
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": "auto",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
    }


def main(folder):
    model = YOLO(MODEL_PATH)
    with open(CLASS_FILE, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    # правила нужны только для тех классов, которых модель сама ещё не знает
    rules = {label: rule for label, rule in RULES.items()
             if label in classes and label not in model.names.values()}
    for label in rules:
        print(f"Модель пока не знает '{label}' — размечаю его правилом")

    images = [f for f in sorted(os.listdir(folder))
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not images:
        raise SystemExit(f"В папке {folder} нет картинок")

    done = skipped = 0
    for name in images:
        img_path = os.path.join(folder, name)
        json_path = os.path.join(folder, os.path.splitext(name)[0] + ".json")
        if os.path.exists(json_path) and not OVERWRITE:
            skipped += 1          # уже размечено (возможно, вручную) — не трогаем
            continue

        img = Image.open(img_path)
        width, height = img.size
        result = model(img_path, conf=CONFIDENCE, verbose=False)[0]

        shapes = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            label = model.names[int(box.cls[0])]
            shapes.append(make_shape(label, x1, y1, x2, y2, float(box.conf[0])))

        for label, rule in rules.items():
            for x1, y1, x2, y2 in rule(img):
                shapes.append(make_shape(label, x1, y1, x2, y2, 1.0))

        data = {
            "version": "4.1.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        done += 1
        found = ", ".join(s["label"] for s in shapes) or "ничего"
        print(f"[{done}] {name}: {found}")

    print(f"\nГотово. Размечено: {done}, пропущено (уже был .json): {skipped}")
    print("Теперь открой папку в X-AnyLabeling и проверь рамки.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1] if len(sys.argv) > 1 else "screenshots")

"""
train.py — обучение модели YOLO (Задача 3).

Запуск:  python train.py
Результат: runs/detect/cookie_detector/weights/best.pt
"""

import os

from ultralytics import YOLO

if __name__ == '__main__':
    # Переходим в папку проекта: тогда data.yaml и папка runs
    # находятся правильно, даже если запускать из другой папки
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. Загрузка предобученной модели
    model = YOLO('yolov8n.pt')

    # 2. Запуск обучения
    results = model.train(
        data='data.yaml',
        epochs=100,        # количество эпох
        imgsz=640,         # размер изображения
        batch=16,
        workers=0,         # отключаем многопоточку (много проблем с этой фигней)
        name='cookie_detector',
        exist_ok=True,     # при повторном обучении перезаписывать папку cookie_detector,
                           # а не создавать cookie_detector2 (иначе app.py не найдёт best.pt)
        patience=50,       # остановка, если нет улучшений 50 эпох подряд
    )

"""
screener.py — автоматический сбор данных (Задача 5 / пункт 1 отчёта).

Каждые INTERVAL секунд делает скриншот экрана и сохраняет его в папку screenshots.
Улучшение по сравнению с методичкой: если экран почти не изменился
с прошлого снимка, скриншот НЕ сохраняется — так в датасете
не будет 100 одинаковых картинок.

Запуск:  python screener.py
Остановка: Ctrl+C
"""

import os
import time
from datetime import datetime

import numpy as np
import pyautogui

save_dir = "screenshots"   # папка для сохранения скриншотов
INTERVAL = 3               # секунд между снимками
MIN_CHANGE = 2.0           # насколько должен измениться экран (средняя разница пикселей, 0-255)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(save_dir, exist_ok=True)


def small_gray(img):
    """Уменьшенная ч/б копия скриншота — чтобы быстро сравнивать кадры."""
    return np.asarray(img.convert("L").resize((160, 90)), dtype=np.float32)


print(f"Начинаю делать скриншоты каждые {INTERVAL} секунд...")
print("Играй в игру: покупай улучшения, жди золотые печеньки и рекламу.")
print("Нажмите Ctrl+C для остановки\n")

count = 0
skipped = 0
prev = None

try:
    while True:
        # Делаем скриншот
        screenshot = pyautogui.screenshot()
        cur = small_gray(screenshot)

        # Пропускаем, если экран почти не поменялся
        if prev is not None and np.abs(cur - prev).mean() < MIN_CHANGE:
            skipped += 1
        else:
            # Формируем имя файла с датой и временем
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"{save_dir}/screenshot_{timestamp}.png"

            # Сохраняем
            screenshot.save(filename)
            count += 1
            prev = cur
            print(f"[{count}] Сохранено: {filename}")

        # Ждём перед следующим снимком
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print(f"\nОстановлено. Сохранено скриншотов: {count}, пропущено одинаковых: {skipped}")

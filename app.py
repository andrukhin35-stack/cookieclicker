"""Cookie Clicker: continuous clicks, YOLOv8 ONNX + OCR. SPACE to stop."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from vision import ShopScanner, CookieTracker, find_ad_buttons

ROOT = Path(__file__).resolve().parent
CLASSES = ("game", "close_ad", "upgrade")
LOG = logging.getLogger("cookie")


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    # Accept old configurations; live clicking now uses clicks_per_second.
    config.setdefault('click_interval',.001)
    config.setdefault('clicks_per_round',30)
    config.setdefault('max_burst_seconds',.10)
    config.setdefault('ad_guard_enabled',False)
    classes = config["classes"]
    if not isinstance(classes, list) or not classes or any(not isinstance(n, str) or not n for n in classes) or len(classes) != len(set(classes)):
        raise ValueError("classes must contain unique model class names in model order")
    if set(config["class_actions"]) != set(classes) or any(n not in CLASSES for n in config["class_actions"].values()):
        raise ValueError("class_actions must map every model class to a known action")
    limits = {"input_size": (640, 640), "iou_threshold": (0.01, 1),
              "loop_interval": (0.005, 10), "startup_delay": (0, 60),
              "click_interval": (0.001, 10), "clicks_per_round": (1, 100),
              "max_burst_seconds": (0.02, 0.5),
              "ad_settle_seconds": (0.05, 5),
              "max_inference_age": (0.05, 10)}
    config.setdefault('analysis_interval',.25)
    config.setdefault('target_ttl',.50)
    config.setdefault('clicks_per_second',250)
    limits.update({'analysis_interval':(.05,5),'target_ttl':(.1,2),'clicks_per_second':(1,1000)})
    for key, (low, high) in limits.items():
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid {key}: expected {low}..{high}")
    if not isinstance(config["clicks_per_round"], int):
        raise ValueError("clicks_per_round must be an integer")
    for key in ("confidence", "cooldowns"):
        expected = classes if key == "confidence" else CLASSES
        if set(config[key]) != set(expected):
            raise ValueError(f"{key}: expected keys {expected}")
        for name, value in config[key].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {key}.{name}")
            if not (0.01 <= value <= 1 if key == "confidence" else 0 <= value <= 60):
                raise ValueError(f"Invalid {key}.{name}")
    region = config["region"]
    if region is not None and (not isinstance(region, list) or len(region) != 4 or
            any(type(v) is not int for v in region) or min(region[:2]) < 0 or min(region[2:]) <= 0):
        raise ValueError("region must be null or [left, top, width, height]")
    if type(config["require_game_for_upgrades"]) is not bool:
        raise ValueError("require_game_for_upgrades must be true or false")
    if type(config["ad_guard_enabled"]) is not bool:
        raise ValueError("ad_guard_enabled must be true or false")
    if type(config['ad_auto_close']) is not bool:
        raise ValueError('ad_auto_close must be true or false')
    shopping = config['shopping']
    if type(shopping['enabled']) is not bool:
        raise ValueError('shopping.enabled must be true or false')
    for key,low,high in [('ocr_confidence',.1,1),
                         ('bottom_fraction',.3,1),('settle_seconds',.05,5),('failed_retry_seconds',.5,30)]:
        value = shopping[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Invalid shopping.{key}')
    column = shopping['price_column']
    if not isinstance(column,list) or len(column)!=2 or any(type(v) not in (float,int) for v in column) or not 0 <= column[0] < column[1] <= 1:
        raise ValueError('shopping.price_column must be [left_fraction, right_fraction]')
    shop_region = shopping['region']
    if shop_region is not None and (not isinstance(shop_region,list) or len(shop_region)!=4 or any(type(v) is not int for v in shop_region) or min(shop_region[:2])<0 or min(shop_region[2:])<=0):
        raise ValueError('shopping.region must be null or [left, top, width, height]')
    model = (path.parent / config["model"]).resolve()
    if not model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model}")
    config["model"] = str(model)
    return config


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    box: tuple[float, float, float, float]
    price: float = 0.0
    source: str = 'onnx'

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class Detector:
    def __init__(self, config):
        self.config = config
        import onnxruntime as ort
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(Path(config['model']).read_bytes(),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name

    def detect(self, image, include_shop_hints=False):
        """BGR image -> classwise NMS -> boxes in original image pixels."""
        height, width = image.shape[:2]
        size = int(self.config["input_size"])
        ratio = min(size / width, size / height)
        resized_w, resized_h = round(width * ratio), round(height * ratio)
        left = round((size - resized_w) / 2 - 0.1)
        top = round((size - resized_h) / 2 - 0.1)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[top:top + resized_h, left:left + resized_w] = cv2.resize(image, (resized_w, resized_h))
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (size, size), swapRB=True)
        output = self.session.run(None,{self.input_name:blob})[0]
        channels = 4 + len(self.config["classes"])
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != channels:
            raise ValueError(f"Expected YOLOv8 output [1, {channels}, N], got {output.shape}")
        rows = output[0].T
        rows = rows[np.isfinite(rows).all(axis=1)]
        ids = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), ids + 4]
        found = []
        for class_id, name in enumerate(self.config["classes"]):
            action = self.config['class_actions'][name]
            threshold = self.config['confidence'][name]
            mask = (ids == class_id) & (scores >= threshold)
            boxes, confidences = [], []
            for row, score in zip(rows[mask], scores[mask]):
                cx, cy, bw, bh = row[:4]
                if bw <= 0 or bh <= 0:
                    continue
                x1 = float(np.clip((cx - bw / 2 - left) / ratio, 0, width - 1))
                y1 = float(np.clip((cy - bh / 2 - top) / ratio, 0, height - 1))
                x2 = float(np.clip((cx + bw / 2 - left) / ratio, 0, width - 1))
                y2 = float(np.clip((cy + bh / 2 - top) / ratio, 0, height - 1))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(float(score))
            if not boxes:
                continue
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, self.config["iou_threshold"])
            for i in np.asarray(indices).reshape(-1):
                x, y, bw, bh = boxes[i]
                kind = 'shop_hint' if include_shop_hints and action=='upgrade' else action
                found.append(Detection(kind, confidences[i], (x, y, x + bw, y + bh)))
        return sorted(found, key=lambda d: d.confidence, reverse=True)


class Controller:
    """One priority action per fresh frame; monotonic class cooldowns."""
    def __init__(self, config, stop):
        self.config, self.stop = config, stop
        self.last = {name: -math.inf for name in CLASSES}
        self.stats = dict.fromkeys(CLASSES, 0)

    def choose(self, detections, now):
        groups = {name: [d for d in detections if d.name == name] for name in CLASSES}
        if any(d.name == "ad_block" for d in detections) and not groups["close_ad"]:
            return None
        # An ad blocks lower priorities even during its cooldown.
        for name in ("close_ad",):
            if groups[name]:
                return max(groups[name], key=lambda d: (d.source=='visual_x',d.confidence)) if self.ready(name, now) else None
        if groups["upgrade"] and (groups["game"] or not self.config["require_game_for_upgrades"]):
            # Spend the budget first; do not earn extra cookies between purchases.
            return max(groups["upgrade"], key=lambda d: (d.price,d.center[1])) if self.ready('upgrade',now) else None
        if groups["game"] and self.ready("game", now):
            return max(groups["game"], key=lambda d: d.confidence)
        return None

    def ready(self, name, now):
        return now - self.last[name] >= self.config["cooldowns"][name]

    def perform(self, detection, click, offset, screen_size, frame_time, dry_run=False):
        count = self.config["clicks_per_round"] if detection.name == "game" else 1
        x, y = detection.center
        x, y = round(x + offset[0]), round(y + offset[1])
        if not (0 <= x < screen_size[0] and 0 <= y < screen_size[1]):
            return 0
        done = 0
        burst_start = time.monotonic()
        for index in range(count):
            if self.stop.is_set() or time.monotonic() - frame_time > self.config["max_inference_age"]:
                break
            if index and time.monotonic()-burst_start >= self.config['max_burst_seconds']:
                break
            # Check immediately before each individual click, including bursts.
            if not dry_run:
                click(x, y)
            self.last[detection.name] = time.monotonic()
            self.stats[detection.name] += 1
            done += 1
            if index + 1 < count and self.stop.wait(self.config["click_interval"]):
                break
        if done:
            LOG.info('%s %s x%d source=%s score=%.3f price=%g at (%d, %d)',
                     'DRY-RUN' if dry_run else 'CLICK',detection.name,done,detection.source,detection.confidence,detection.price,x,y)
        return done


def watch_space(stop, ready=None, errors=None):
    registered = False
    user32 = None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        # The OS queues WM_HOTKEY, so a quick tap cannot fall between polls.
        registered = bool(user32.RegisterHotKey(None, 1, 0x4000, 0x20))
        if not registered:
            raise OSError(ctypes.get_last_error(), "Cannot register SPACE hotkey; run in a normal Windows terminal")
        if ready is not None:
            ready.set()
        message = wintypes.MSG()
        while not stop.is_set():
            while user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 1):
                if message.message == 0x0312 and message.wParam == 1:
                    stop.set()
                    return
            stop.wait(0.02)
    except Exception as exc:
        if errors is not None:
            errors.append(str(exc))
        LOG.error("Keyboard monitoring failed; stopping: %s", exc)
        stop.set()
    finally:
        if registered:
            user32.UnregisterHotKey(None, 1)
        if ready is not None:
            ready.set()


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def visible_ad_close(image):
    """Conservative blocking guard for gray circular X buttons; never clicks."""
    pixels = image.astype(np.int16)
    b, g, r = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    gray = ((abs(r - g) < 12) & (abs(g - b) < 16) & (r > 80) & (r < 160)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(gray, 8)
    white = pixels.min(axis=2) > 200
    for x, y, w, h, area in stats[1:count]:
        if 24 <= w <= 96 and abs(w - h) <= max(4, w * .12) and area >= .4 * w * h:
            center = white[y + h//4:y + 3*h//4, x + w//4:x + 3*w//4]
            if center.size and center.mean() > .12:
                return True
    return False


def analyze_frame(detector, scanner, image, config, cookie=None):
    # Closing ads must never wait behind ONNX or OCR.
    buttons = find_ad_buttons(image) if config['ad_auto_close'] else []
    if buttons:
        return [Detection('close_ad',.99,box,source='visual_x') for box in buttons]
    raw = detector.detect(image, include_shop_hints=True)
    found = [d for d in raw if d.name != 'shop_hint']
    if cookie is not None:
        found = [d for d in found if d.name!='game'] + [cookie]
    # Never scan prices or make purchase decisions on top of an ad.
    if any(d.name in ('close_ad','ad_block') for d in found):
        return found
    if config['require_game_for_upgrades'] and not any(d.name=='game' for d in found):
        return found
    offers = scanner.scan(image,[d for d in raw if d.name=='shop_hint'])
    found.extend(Detection('upgrade',o.confidence,o.box,o.price,'green_price_ocr') for o in offers)
    return found


def unchanged_patch(before, after, box):
    """Reject an asynchronous offer/close target if its pixels have changed."""
    x1,y1,x2,y2 = map(round,box)
    if before.shape!=after.shape or min(x1,y1)<0 or x2>after.shape[1] or y2>after.shape[0]:
        return False
    a,b = before[y1:y2,x1:x2],after[y1:y2,x1:x2]
    return bool(a.size and np.abs(a.astype(np.int16)-b.astype(np.int16)).mean()<4)


class ClickStream:
    """One mouse owner. Continuous clicks expire unless fresh vision renews them."""
    def __init__(self,config,stop,click,screen,offset=(0,0),dry_run=False):
        self.config,self.stop,self.click = config,stop,click
        self.screen,self.offset,self.dry_run = screen,offset,dry_run
        self.lock = threading.Lock()
        self.target = None
        self.expires = 0
        self.stats = dict.fromkeys(CLASSES,0)
        self.thread = threading.Thread(target=self.run,daemon=True,name='cookie-clicks')
        self.error = None

    def publish(self,target,frame_time):
        with self.lock:
            self.target = target
            self.expires = frame_time+self.config['target_ttl']

    def _click(self,target):
        x,y = target.center
        x,y = round(x+self.offset[0]),round(y+self.offset[1])
        if not (0<=x<self.screen[0] and 0<=y<self.screen[1]):
            return False
        if not self.dry_run:
            self.click(x,y)
        self.stats[target.name]+=1
        return True

    def action(self,target):
        # The same lock prevents a cookie click between moving to and clicking
        # a product or an ad. Resume only after a new capture confirms the cookie.
        with self.lock:
            self.target = None
            if self.stop.is_set():
                return False
            done = self._click(target)
        if done:
            LOG.info('%s %s price=%g source=%s at %s',
                     'DRY-RUN' if self.dry_run else 'CLICK',target.name,target.price,target.source,target.center)
        return done

    def run(self):
        period = max(1/self.config['clicks_per_second'],self.config['cooldowns']['game'])
        try:
            while not self.stop.is_set():
                started = time.perf_counter()
                with self.lock:
                    if self.target is not None and time.monotonic()<self.expires and not self.stop.is_set():
                        self._click(self.target)
                # Python's Windows sleep uses a high resolution timer. Event.wait
                # with 1 ms used to round up and made the old bursts crawl.
                time.sleep(max(.0005,period-(time.perf_counter()-started)))
        except Exception as exc:
            self.error = exc
            LOG.error('Mouse stream stopped: %s',exc)
            self.stop.set()


class BackgroundAnalysis:
    """Latest frame only: inference/OCR cannot build a stale frame backlog."""
    def __init__(self,detector,scanner,config,stop):
        self.detector,self.scanner,self.config,self.stop=detector,scanner,config,stop
        self.condition = threading.Condition()
        self.pending = None
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self.run,daemon=True,name='onnx-ocr')

    def submit(self,image,frame_time,cookie):
        with self.condition:
            self.pending = (image,frame_time,cookie)
            self.condition.notify()

    def run(self):
        try:
            while not self.stop.is_set():
                with self.condition:
                    self.condition.wait_for(lambda:self.pending is not None or self.stop.is_set(),timeout=.1)
                    if self.pending is None:
                        continue
                    image,stamp,cookie = self.pending
                    self.pending = None
                started = time.monotonic()
                found = analyze_frame(self.detector,self.scanner,image,self.config,cookie)
                self.latest = (image,stamp,found,time.monotonic()-started)
        except Exception as exc:
            self.error = exc
            LOG.error('ONNX/OCR stopped: %s',exc)
            self.stop.set()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--dry-run", action="store_true", help="Detect without clicking")
    parser.add_argument("--image", type=Path, help="Offline inference on one image; no mouse or keyboard access")
    parser.add_argument('--seconds',type=float,help='Stop automatically after this many seconds (live test)')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    watcher = None
    controller = None
    keyboard_errors = []
    stream = analysis = None
    try:
        config = load_config(args.config)
        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0):
            raise ValueError('--seconds must be positive')
        LOG.info('Continuous clicker v2 | app=%s | config=%s',Path(__file__).resolve(),args.config.resolve())
        if not args.image:
            if os.name != "nt":
                raise ValueError("Live mode supports Windows. Use --image for offline inference.")
            enable_dpi_awareness()
            keyboard_ready = threading.Event()
            watcher = threading.Thread(target=watch_space, args=(stop, keyboard_ready, keyboard_errors), daemon=True)
            watcher.start()
            if not keyboard_ready.wait(2):
                raise RuntimeError("Keyboard monitoring did not start")
            if keyboard_errors:
                raise RuntimeError(keyboard_errors[0])
            if stop.is_set():
                return 0
        detector = Detector(config)
        scanner = ShopScanner(config)
        tracker = CookieTracker()
        if args.image:
            data = np.fromfile(args.image, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {args.image}")
            match = tracker.find(image)
            cookie = Detection('game',match[0],match[1],source='cookie_texture') if match else None
            print(json.dumps([d.__dict__ for d in analyze_frame(detector,scanner,image,config,cookie)], ensure_ascii=False, indent=2))
            return 0
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0
        screen = tuple(pyautogui.size())
        region = config["region"] or [0, 0, *screen]
        x, y, w, h = region
        if x + w > screen[0] or y + h > screen[1]:
            raise ValueError("region extends beyond the primary monitor")
        controller = Controller(config, stop)
        stream = ClickStream(config,stop,pyautogui.click,screen,(x,y),args.dry_run)
        analysis = BackgroundAnalysis(detector,scanner,config,stop)
        LOG.info("SPACE / Ctrl+C to stop; failsafe at screen corners. Startup in %.1fs.", config["startup_delay"])
        if stop.wait(config["startup_delay"]):
            return 1 if keyboard_errors else 0
        stream.thread.start()
        analysis.thread.start()
        started = time.monotonic()
        last_submit = last_status = last_clicks = 0
        used_analysis = -math.inf
        min_offer_time = -math.inf
        last_game = None
        while not stop.is_set():
            frame_time = time.monotonic()
            if args.seconds is not None and frame_time-started>=args.seconds:
                break
            screenshot = pyautogui.screenshot(region=tuple(region))
            image = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_RGB2BGR)
            buttons = find_ad_buttons(image) if config['ad_auto_close'] else []
            match = None if buttons else tracker.find(image)
            cookie = Detection('game',match[0],match[1],source='cookie_texture') if match else None
            latest = analysis.latest
            offers,neural_ads = [],[]
            now = time.monotonic()
            if latest is not None:
                old,stamp,found,duration = latest
                if now-stamp<=config['max_inference_age']:
                    # A confident YOLO target can bridge texture animation only
                    # while the actual current pixels still match its capture.
                    if cookie is None and not buttons:
                        games=[d for d in found if d.name=='game' and unchanged_patch(old,image,d.box) and tracker.confirm(image,d.box)]
                        cookie=max(games,key=lambda d:d.confidence) if games else None
                    neural_ads=[d for d in found if d.name=='close_ad' and unchanged_patch(old,image,d.box)]
                    if stamp>used_analysis and stamp>=min_offer_time:
                        offers=[d for d in found if d.name=='upgrade' and unchanged_patch(old,image,d.box)]
            ads=[Detection('close_ad',.99,b,source='visual_x') for b in buttons] or neural_ads
            status = 'cookie' if cookie else 'NO COOKIE: game hidden / texture not found'
            if ads:
                stream.publish(None,frame_time)
                status = 'advertisement'
                if controller.ready('close_ad',now):
                    target=max(ads,key=lambda d:d.confidence)
                    if stream.action(target):
                        controller.last['close_ad']=time.monotonic()
                        stop.wait(config['ad_settle_seconds'])
            else:
                if cookie is not None and last_game is None:
                    LOG.info('Cookie acquired: source=%s score=%.3f center=%s',cookie.source,cookie.confidence,cookie.center)
                stream.publish(cookie,frame_time)
                if offers and cookie is not None and controller.ready('upgrade',now):
                    target=max(offers,key=lambda d:(d.price,d.center[1]))
                    if stream.action(target):
                        controller.last['upgrade']=time.monotonic()
                        used_analysis=latest[1]
                        scanner.record_attempt(target)
                        stop.wait(config['shopping']['settle_seconds'])
                        min_offer_time=time.monotonic()
                        # Only a post-purchase frame may authorize another buy.
                        last_submit=0
                        continue
                if now-last_submit>=config['analysis_interval']:
                    # No sleeps or model calls in the mouse stream.
                    analysis.submit(image,frame_time,cookie)
                    last_submit=now
            last_game=cookie
            if now-last_status>=2:
                total=stream.stats['game']
                cps=(total-last_clicks)/(now-last_status) if last_status else 0
                LOG.info('STATUS %s | %.0f clicks/s | total=%s | capture=%.0fms | analysis=%s',
                         status,cps,stream.stats,(now-frame_time)*1000,
                         ('%.0fms'%(latest[3]*1000)) if latest else 'starting')
                last_status,last_clicks=now,total
            stop.wait(config["loop_interval"])
        if stream.error or analysis.error:
            return 1
        return 1 if keyboard_errors else 0
    except KeyboardInterrupt:
        LOG.info("Stopped by Ctrl+C")
        return 0
    except Exception as exc:
        # Any screenshot, model, config or mouse error stops the bot.
        LOG.error("Stopped: %s", exc)
        return 1
    finally:
        stop.set()
        if stream is not None and stream.thread.is_alive():
            stream.thread.join(timeout=1)
            LOG.info('Actions%s: %s',' (simulated)' if args.dry_run else '',stream.stats)
        if watcher is not None:
            watcher.join(timeout=0.2)
        if controller is not None:
            LOG.info("Actions%s: %s", " (simulated)" if args.dry_run else "", controller.stats)


if __name__ == "__main__":
    raise SystemExit(main())

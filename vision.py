"""Visual checks for the game's close buttons and green, affordable prices."""
from dataclasses import dataclass
import hashlib
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np


class CookieTracker:
    """Verify the actual cookie texture on every capture, independent of YOLO/OCR."""
    def __init__(self):
        path = Path(__file__).resolve().parent/'assets/cookie.png'
        self.template = cv2.cvtColor(cv2.imdecode(np.fromfile(path,np.uint8),1),cv2.COLOR_BGR2GRAY)

    def confirm(self,image,box):
        x1,y1,x2,y2=map(round,box)
        crop=image[max(0,y1):y2,max(0,x1):x2]
        if not crop.size or min(crop.shape[:2])<35:
            return False
        b,g,r=cv2.split(crop.astype(np.float32))
        if ((r>80)&(r>g*1.12)&(g>b*1.15)).mean()<.25:
            return False
        gray=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(96,96))
        scores=cv2.matchTemplate(gray,self.template[16:80,16:80],cv2.TM_CCOEFF_NORMED)
        return float(scores[8:25,8:25].max())>=.45

    def find(self, image):
        # Brown connected components propose locations; texture rejects wood,
        # product icons and objects such as the car wheel seen by YOLO.
        scale = min(1.0,1280/image.shape[1])
        small = cv2.resize(image,None,fx=scale,fy=scale) if scale < 1 else image
        b,g,r = cv2.split(small.astype(np.float32))
        brown = ((r>100)&(g>60)&(r>g*1.12)&(g>b*1.15)).astype(np.uint8)
        brown = cv2.morphologyEx(brown,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        brown = cv2.morphologyEx(brown,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        contours,_ = cv2.findContours(brown,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if min(w,h)<35 or max(w,h)>550*scale or not .78<w/h<1.25:
                continue
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour,True)
            if area/(w*h)<.45 or 4*math.pi*area/max(1,perimeter**2)<.40:
                continue
            # The bright interior can be smaller than the dark baked rim.
            score = -1
            for expansion in (1.0,1.08,1.16):
                dw,dh = round(w*(expansion-1)/2),round(h*(expansion-1)/2)
                x0,y0,x3,y3 = max(0,x-dw),max(0,y-dh),min(small.shape[1],x+w+dw),min(small.shape[0],y+h+dh)
                gray = cv2.resize(cv2.cvtColor(small[y0:y3,x0:x3],cv2.COLOR_BGR2GRAY),(96,96))
                # Ignore the animated rim, cursors and +1 labels. Match the
                # distinctive central chocolate chips with limited translation.
                scores = cv2.matchTemplate(gray,self.template[16:80,16:80],cv2.TM_CCOEFF_NORMED)
                score = max(score,float(scores[8:25,8:25].max()))
            if score>=.64 and (best is None or score>best[0]):
                best = (score,(x/scale,y/scale,(x+w)/scale,(y+h)/scale))
        return best


def find_ad_buttons(image):
    """Gray circular button + white diagonal X. No hardcoded click position."""
    pixels = image.astype(np.int16)
    b, g, r = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    mask = ((abs(r-g) < 16) & (abs(g-b) < 20) & (r > 65) & (r < 205)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    white = pixels.min(axis=2) > 195
    found = []
    for ident in range(1, count):
        x, y, w, h, area = stats[ident]
        if not (14 <= w <= 180 and abs(w-h) <= max(4, w*.16) and .30 <= area/(w*h) <= .90):
            continue
        component = labels[y:y+h, x:x+w] == ident
        k = max(2, round(w*.12))
        corners = np.concatenate([component[:k,:k].ravel(),component[:k,-k:].ravel(),
                                  component[-k:,:k].ravel(),component[-k:,-k:].ravel()])
        if corners.mean() > .35:
            continue
        center = white[y+h//5:y+4*h//5, x+w//5:x+4*w//5]
        if not center.size or not .10 <= center.mean() <= .65:
            continue
        small = cv2.resize(center.astype(np.uint8), (32,32), interpolation=cv2.INTER_NEAREST).astype(bool)
        yy, xx = np.indices(small.shape)
        diagonal = (abs(xx-yy) <= 5) | (abs(xx+yy-31) <= 5)
        if (small & diagonal).sum() / max(1, small.sum()) < .82:
            continue
        if any(small[a:a+16,b:b+16].sum() < 5 for a,b in ((0,0),(0,16),(16,0),(16,16))):
            continue
        found.append((float(x),float(y),float(x+w),float(y+h)))
    return found


def parse_price(text):
    text = text.casefold().strip().replace('\u00a0','').replace('\u202f','').replace(' ','')
    suffixes = {'':1, 'k':1e3, 'тыс':1e3, 'тыс.':1e3, 'thousand':1e3,
                'тысяча':1e3, 'тысячи':1e3, 'тысяч':1e3,
                'm':1e6, 'млн':1e6, 'млн.':1e6, 'million':1e6,
                'миллион':1e6, 'миллиона':1e6, 'миллионов':1e6,
                'b':1e9, 'млрд':1e9, 'млрд.':1e9, 'billion':1e9,
                'миллиард':1e9, 'миллиарда':1e9, 'миллиардов':1e9,
                't':1e12, 'трлн':1e12, 'трлн.':1e12, 'trillion':1e12,
                'триллион':1e12, 'триллиона':1e12, 'триллионов':1e12,
                'quadrillion':1e15, 'quintillion':1e18, 'sextillion':1e21,
                'septillion':1e24, 'octillion':1e27, 'nonillion':1e30, 'decillion':1e33}
    match = re.fullmatch(r'([0-9][0-9.,]*)([^0-9]*)', text)
    if not match or match[2] not in suffixes:
        return None
    number, suffix = match.groups()
    if suffix:
        if number.count(',') + number.count('.') > 1:
            return None
        number = number.replace(',','.')
    elif re.fullmatch(r'\d{1,3}([,.]\d{3})+',number):
        number = number.replace(',','').replace('.','')
    elif ',' in number or '.' in number:
        return None
    try:
        price = float(number)*suffixes[suffix]
    except ValueError:
        return None
    return price if math.isfinite(price) and price > 0 else None


@dataclass(frozen=True)
class Offer:
    price: float
    box: tuple
    text: str
    confidence: float


class ShopScanner:
    def __init__(self, config):
        self.config = config
        self.reader = None
        self.cache = {}
        self.attempts = {}

    def read_prices(self, crops):
        if self.reader is None:
            import onnxruntime as ort
            model = Path(__file__).resolve().parent/'assets/cyrillic_rec.onnx'
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.reader = ort.InferenceSession(model.read_bytes(),sess_options=options,providers=['CPUExecutionProvider'])
            self.characters = ['blank'] + self.reader.get_modelmeta().custom_metadata_map['character'].splitlines() + [' ']
        height = 48
        widths = [max(16,math.ceil(height*im.shape[1]/im.shape[0])) for im in crops]
        batch = np.zeros((len(crops),3,height,max(widths)),np.float32)
        for index,(image,width) in enumerate(zip(crops,widths)):
            resized = cv2.resize(image,(width,height)).astype(np.float32)/127.5-1
            batch[index,:,:,:width] = resized.transpose(2,0,1)
        output = self.reader.run(None,{self.reader.get_inputs()[0].name:batch})[0]
        if output.shape[-1] != len(self.characters):
            raise ValueError('OCR alphabet does not match the recognition model')
        values = []
        for row in output:
            indices, probabilities = row.argmax(axis=1), row.max(axis=1)
            chars, scores, previous = [], [], -1
            for index, probability in zip(indices,probabilities):
                if index and index != previous:
                    chars.append(self.characters[index])
                    scores.append(float(probability))
                previous = index
            values.append((''.join(chars),float(np.mean(scores)) if scores else 0.0))
        return values

    def scan(self, image, hints):
        settings = self.config['shopping']
        if not settings['enabled']:
            return []
        height, width = image.shape[:2]
        if settings['region'] is not None:
            x, y, w, h = settings['region']
            x1,y1,x2,y2 = x,y,x+w,y+h
        else:
            hints = [d for d in hints if d.box[2]-d.box[0] >= width*.08 and d.center[0] > width*.40]
            if not hints:
                return []
            anchor = max(hints, key=lambda d:d.confidence)
            x1,y1,x2,_ = anchor.box
            y2 = height * settings['bottom_fraction']
        x1,y1,x2,y2 = max(0,int(x1)),max(0,int(y1)),min(width,int(x2)),min(height,int(y2))
        if x2 <= x1 or y2 <= y1:
            return []
        panel_w = x2-x1
        left = x1 + int(panel_w*settings['price_column'][0])
        right = x1 + int(panel_w*settings['price_column'][1])
        rgb = image[y1:y2,left:right].astype(np.int16)
        if not rgb.size:
            return []
        b,g,r = rgb[...,0],rgb[...,1],rgb[...,2]
        green = (g > 145) & (g > r*1.4) & (g > b*1.35)
        rows = (green.sum(axis=1) >= 3).astype(np.uint8)[:,None]
        rows = cv2.morphologyEx(rows,cv2.MORPH_CLOSE,np.ones((4,1),np.uint8))
        count,_,stats,_ = cv2.connectedComponentsWithStats(rows,8)
        candidates = []
        for _,top,_,text_h,_ in stats[1:count]:
            if not 5 <= text_h <= 42:
                continue
            xs = np.where(green[top:top+text_h].any(axis=0))[0]
            if not len(xs) or xs[-1]-xs[0] < 3:
                continue
            # Green selects the affordable row; preserve the original glyphs and
            # antialiasing for OCR rather than damaging small digits with a mask.
            crop = image[max(0,y1+top-3):min(height,y1+top+text_h+3),
                         max(0,left+xs[0]-3):min(width,left+xs[-1]+4)].copy()
            crop = cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
            letters = green[top:top+text_h,xs[0]:xs[-1]+1]
            binary = np.full((text_h+12,letters.shape[1]+12,3),255,np.uint8)
            binary[6:6+text_h,6:6+letters.shape[1]][letters] = 0
            binary = cv2.resize(binary,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
            key = hashlib.sha1(crop.tobytes()).hexdigest()
            box = (float(left+xs[0]),float(y1+top),float(left+xs[-1]+1),float(y1+top+text_h))
            candidates.append((key,(crop,binary),box))
        missing = {key:variants for key,variants,_ in candidates if key not in self.cache}
        if missing:
            results = self.read_prices([image for variants in missing.values() for image in variants])
            if len(self.cache) > 512:
                self.cache.clear()
            for index,key in enumerate(missing):
                valid = [(t,c) for t,c in results[2*index:2*index+2] if parse_price(t) is not None and c >= settings['ocr_confidence']]
                # A recognized unit must not be discarded in favor of digits alone.
                self.cache[key] = max(valid,key=lambda r:(any(ch.isalpha() for ch in r[0]),r[1])) if valid else ('',0)
        offers = []
        now = time.monotonic()
        for key,_,box in candidates:
            text,confidence = self.cache[key]
            price = parse_price(text)
            if price is None or float(confidence) < settings['ocr_confidence']:
                continue
            offer = Offer(price,box,text,float(confidence))
            attempts,blocked = self.attempts.get(self.key(offer),(0,0))
            if now >= blocked:
                offers.append(offer)
        return offers

    @staticmethod
    def key(offer):
        return (round((offer.box[0]+offer.box[2])/20),round((offer.box[1]+offer.box[3])/20),offer.price)

    def record_attempt(self, offer):
        key = self.key(offer)
        count,_ = self.attempts.get(key,(0,0))
        count += 1
        if len(self.attempts) > 512:
            self.attempts.clear()
        # An unchanged price after repeated clicks usually means wrong buy/sell/bulk mode.
        blocked = time.monotonic()+self.config['shopping']['failed_retry_seconds'] if count >= 2 else 0
        self.attempts[key] = (0 if blocked else count,blocked)

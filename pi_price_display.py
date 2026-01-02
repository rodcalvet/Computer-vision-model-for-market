import time, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Try Picamera2 first (Pi camera). Fall back to OpenCV webcam.
USE_PICAMERA2 = True
try:
    from picamera2 import Picamera2
except Exception:
    USE_PICAMERA2 = False

import cv2
import tflite_runtime.interpreter as tflite

MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"
PRICES_PATH = "prices.json"

CONF_TH = 0.75          # only accept confident predictions
STABLE_FRAMES = 6       # require same label N frames in a row
COOLDOWN_S = 1.0        # minimum time between changing displayed product

# Display settings
FULLSCREEN = True
WINDOW_NAME = "Price Screen"

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def draw_price_screen(product, price, conf):
    # Create a clean UI image
    W, H = 1024, 600  # you can change this to match your display
    img = Image.new("RGB", (W, H), (15, 15, 18))
    d = ImageDraw.Draw(img)

    # Fonts (fallback if system font not found)
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except:
        font_big = font_med = font_small = ImageFont.load_default()

    title = product.replace("_", " ").upper()
    price_str = f"${price:,.2f}"

    # Center title
    tw, th = d.textbbox((0,0), title, font=font_med)[2:]
    d.text(((W - tw)//2, 110), title, font=font_med, fill=(235, 235, 240))

    # Center price
    pw, ph = d.textbbox((0,0), price_str, font=font_big)[2:]
    d.text(((W - pw)//2, 240), price_str, font=font_big, fill=(120, 255, 140))

    # Confidence small
    conf_str = f"Confidence: {conf*100:.1f}%"
    cw, ch = d.textbbox((0,0), conf_str, font=font_small)[2:]
    d.text(((W - cw)//2, 520), conf_str, font=font_small, fill=(160, 160, 170))

    # Convert PIL -> OpenCV BGR
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def main():
    labels = load_labels(LABELS_PATH)
    prices = json.load(open(PRICES_PATH, "r", encoding="utf-8"))

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_h, in_w = input_details[0]["shape"][1], input_details[0]["shape"][2]
    in_dtype = input_details[0]["dtype"]

    # Camera init
    if USE_PICAMERA2:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
        picam2.start()
        get_frame = lambda: picam2.capture_array()  # RGB
    else:
        cap = cv2.VideoCapture(0)
        def get_frame():
            ok, frame = cap.read()
            if not ok:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB

    # UI window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_label = None
    stable_count = 0
    shown_label = None
    last_change_t = 0

    # Start with an idle screen
    screen = draw_price_screen("Place product", 0.00, 0.0)
    cv2.imshow(WINDOW_NAME, screen)

    while True:
        rgb = get_frame()
        if rgb is None:
            break

        # Crop center square (helps for “one product at a time”)
        h, w, _ = rgb.shape
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        crop = rgb[y0:y0+side, x0:x0+side]

        # Resize to model input
        img = cv2.resize(crop, (in_w, in_h), interpolation=cv2.INTER_AREA)

        # Normalize / dtype handling
        if in_dtype == np.float32:
            inp = (img.astype(np.float32) / 255.0)[None, ...]
        else:
            # uint8 models expect 0..255
            inp = img.astype(np.uint8)[None, ...]

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])[0]

        # Some models output logits, some probabilities
        probs = softmax(out) if np.max(out) > 1.0 else out
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        label = labels[cls] if cls < len(labels) else f"class_{cls}"

        # Stabilize prediction (avoid flicker)
        if label == last_label and conf >= CONF_TH:
            stable_count += 1
        else:
            stable_count = 0
        last_label = label

        now = time.time()
        if stable_count >= STABLE_FRAMES and (now - last_change_t) >= COOLDOWN_S:
            if label in prices and label != shown_label:
                shown_label = label
                last_change_t = now
                screen = draw_price_screen(label, float(prices[label]), conf)
                cv2.imshow(WINDOW_NAME, screen)

        # ESC to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


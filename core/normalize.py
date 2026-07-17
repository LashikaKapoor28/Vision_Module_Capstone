import cv2
import numpy as np
from pathlib import Path

def resize_images(folder_path, max_width=1280):
    for path in Path(folder_path).rglob('*'):
        if path.is_file():
            try:
                buf = np.fromfile(str(path), dtype=np.uint8)
            except OSError:
                continue
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                if w > max_width:
                    new_h = int(h * (max_width / w))
                    img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
                    ok, encoded = cv2.imencode(path.suffix, img)
                    if ok:
                        encoded.tofile(str(path))

if __name__ == "__main__":
    resize_images("data/known_faces")
    resize_images("data/test_images")

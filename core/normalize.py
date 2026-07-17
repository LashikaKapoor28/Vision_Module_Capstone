import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps


def load_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.asarray(img)


def resize_image(img, max_width=1280):
    h, w = img.shape[:2]
    if w <= max_width:
        return img

    new_h = int(h * (max_width / w))
    return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)


def resize_images(folder_path, max_width=1280):
    for path in Path(folder_path).rglob('*'):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = load_image(path)
        resized = resize_image(img, max_width)
        if resized is not img:
            cv2.imwrite(str(path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    resize_images("data/known_faces")
    resize_images("data/test_images")

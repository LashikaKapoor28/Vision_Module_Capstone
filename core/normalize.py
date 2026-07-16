import cv2
from pathlib import Path


def resize_image(img, max_width=1280):
    h, w = img.shape[:2]
    if w <= max_width:
        return img

    new_h = int(h * (max_width / w))
    return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)


def resize_images(folder_path, max_width=1280):
    for path in Path(folder_path).rglob('*'):
        if path.is_file():
            img = cv2.imread(str(path))
            if img is not None:
                resized = resize_image(img, max_width)
                if resized is not img:
                    img = resized
                    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    resize_images("data/known_faces")
    resize_images("data/test_images")

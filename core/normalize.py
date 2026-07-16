import cv2
from pathlib import Path

def resize_images(folder_path, max_width=1280):
    for path in Path(folder_path).rglob('*'):
        if path.is_file():
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                if w > max_width:
                    new_h = int(h * (max_width / w))
                    img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(path), img)

if __name__ == "__main__":
    resize_images("data/known_faces")
    resize_images("data/test_images")

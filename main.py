# Need to build the video lloop

import sys
from pathlib import Path
from camera import take_picture
from facenet_models import FacenetModel
from core.database import FaceDatabase
from core.normalize import load_image
from recognizer import recognize
from visualize import display_matches

DB_PATH = Path("data/profiles.pkl")
DETECTION_THRESHOLD = 0.87

def load_database():
    if DB_PATH.exists():
        return FaceDatabase.load(DB_PATH)
    return FaceDatabase()


def add_face(image, database, model):
    name = input("Unknown face, enter a name or leave blank to skip: ").strip()
    if not name:
        return
    database.add_image(name, image, model, detection_threshold=DETECTION_THRESHOLD)
    database.save(DB_PATH)


def recognize_face(image, database, model):
    boxes, names = recognize(image, database, model)
    display_matches(image, boxes, names)

    if names and "Unknown" in names:
        add_face(image, database, model)
    else:
        database.save(DB_PATH)

def get_image():
    if len(sys.argv) > 1:
        return load_image(sys.argv[1])
    return take_picture()


def main():
    model = FacenetModel()
    database = load_database()
    image = get_image()
    recognize_face(image, database, model)

if __name__ == "__main__":
    main()


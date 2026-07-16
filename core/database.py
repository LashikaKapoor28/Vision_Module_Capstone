import pickle
from pathlib import Path

import numpy as np

from .profile import Profile


class FaceDatabase:
    def __init__(self, profiles=None):
        if profiles is None:
            profiles = {}

        self.profiles = profiles

    def __len__(self):
        return len(self.profiles)

    def __contains__(self, name):
        return name in self.profiles

    def get_profile(self, name):
        return self.profiles.get(name)

    def add_profile(self, profile):
        if not isinstance(profile, Profile):
            raise TypeError("profile must be a Profile")

        if profile.name in self.profiles:
            raise ValueError(f"{profile.name} is already in the database")

        self.profiles[profile.name] = profile

    def remove_profile(self, name):
        if name not in self.profiles:
            raise KeyError(f"{name} is not in the database")

        return self.profiles.pop(name)

    def add_descriptor(self, name, descriptor):
        name = name.strip()
        descriptor = np.asarray(descriptor)

        if name == "":
            raise ValueError("name cannot be empty")

        if descriptor.shape != (512,):
            raise ValueError("descriptor must have shape (512,)")

        if name not in self.profiles:
            self.profiles[name] = Profile(name, [descriptor])
        else:
            self.profiles[name].add_descriptor(descriptor)

        return self.profiles[name]

    def add_image(self, name, image, model, detection_threshold=0.9):
        image = np.asarray(image)

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must have shape (height, width, 3)")

        boxes, probabilities, landmarks = model.detect(image)

        if boxes is None or probabilities is None:
            raise ValueError("No face was detected")

        boxes = np.asarray(boxes)
        probabilities = np.asarray(probabilities)

        # only keep detections above the probability cutoff
        accepted_boxes = boxes[probabilities >= detection_threshold]

        if len(accepted_boxes) == 0:
            raise ValueError("No face passed the detection threshold")

        if len(accepted_boxes) > 1:
            raise ValueError("The image should contain only one face")

        descriptors = np.asarray(
            model.compute_descriptors(image, accepted_boxes)
        )

        if descriptors.shape == (1, 512):
            descriptor = descriptors[0]
        elif descriptors.shape == (512,):
            descriptor = descriptors
        else:
            raise ValueError("Expected one face descriptor")

        return self.add_descriptor(name, descriptor)

    def save(self, filepath):
        path = Path(filepath)

        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath):
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")

        with open(path, "rb") as file:
            database = pickle.load(file)

        if not isinstance(database, cls):
            raise TypeError("File does not contain a FaceDatabase")

        return database
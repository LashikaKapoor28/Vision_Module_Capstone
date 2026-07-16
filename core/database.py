import numpy as np

from .profile import Profile


class FaceDatabase:
    """Stores profiles using names as dictionary keys."""

    def __init__(self, profiles=None):
        self.profiles = profiles if profiles is not None else {}

    def __len__(self):
        return len(self.profiles)

    def __contains__(self, name):
        return name in self.profiles

    def get_profile(self, name):
        return self.profiles.get(name)

    def add_profile(self, profile):
        if not isinstance(profile, Profile):
            raise TypeError("profile must be a Profile object")

        if profile.name in self.profiles:
            raise ValueError(f"A profile named {profile.name} already exists")

        self.profiles[profile.name] = profile

    def remove_profile(self, name):
        if name not in self.profiles:
            raise KeyError(f"No profile named {name} exists")

        return self.profiles.pop(name)

    def add_descriptor(self, name, descriptor):
        descriptor = np.asarray(descriptor)

        if not isinstance(name, str) or name.strip() == "":
            raise ValueError("name must be a non-empty string")

        if descriptor.shape != (512,):
            raise ValueError("descriptor must have shape (512,)")

        name = name.strip()

        if name not in self.profiles:
            self.profiles[name] = Profile(name, [descriptor])
        else:
            self.profiles[name].add_descriptor(descriptor)

        return self.profiles[name]

    def add_image(self, name, image, model, detection_threshold=0.90):
        image = np.asarray(image)

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must have shape (height, width, 3)")

        boxes, probabilities, _ = model.detect(image)

        if boxes is None or probabilities is None:
            raise ValueError("No face was detected")

        boxes = np.asarray(boxes)
        probabilities = np.asarray(probabilities)

        accepted_boxes = boxes[probabilities >= detection_threshold]

        if len(accepted_boxes) == 0:
            raise ValueError("No face passed the detection threshold")

        if len(accepted_boxes) > 1:
            raise ValueError("The image must contain only one face")

        descriptors = np.asarray(
            model.compute_descriptors(image, accepted_boxes)
        )

        if descriptors.shape == (1, 512):
            descriptor = descriptors[0]
        elif descriptors.shape == (512,):
            descriptor = descriptors
        else:
            raise ValueError("Expected exactly one face descriptor")

        return self.add_descriptor(name, descriptor)
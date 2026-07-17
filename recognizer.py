from core.similarity import cosine_distances
import numpy as np

RECOGNITION_THRESHOLD = 0.2846

def recognize(image, database, model):
      boxes, probabilities, landmarks = model.detect(image)
      descriptors = np.asarray(model.compute_descriptors(image, boxes))
      
      profile_names = list(database.profiles.keys())
      known_descriptors = np.array([database.profiles[name].average_descriptor for name in profile_names])
      
      distances = cosine_distances(descriptors, known_descriptors)
      names = []

      for face_distances in distances:
            best_match = np.argmin(face_distances)

            if face_distances[best_match] < RECOGNITION_THRESHOLD:
                  names.append(profile_names[best_match])
            else:
                  names.append("Unknown")

      return boxes, names
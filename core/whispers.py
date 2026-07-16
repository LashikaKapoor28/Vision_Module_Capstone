import random
import shutil
from facenet_models import FacenetModel
import numpy as np
from skimage import io
from core.similarity import cosine_distances
from core.normalize import resize_images
detection_prob_threshold = 0.87
model = FacenetModel()
class Node:
    def __init__(self, image_path, descriptor, label):
        self.image_path = image_path
        self.descriptor = descriptor
        self.label = label

def get_descriptor(image_path):
    image = io.imread(str(image_path))
    if image.shape[-1] == 4:
        image = image[..., :3]  
    boxes, probabilities, landmarks = model.detect(image)
    if boxes is not None:
        mask = probabilities > detection_prob_threshold
        boxes = boxes[mask]
        probabilities = probabilities[mask]
    if boxes is None or len(boxes) == 0:
        raise ValueError(f"No face detected in {image_path}")

    best = np.argmax(probabilities)
    descriptors = model.compute_descriptors(image, boxes)
    return descriptors[best]


def adj_list(image_paths, threshold):
    for folder in {p.parent for p in image_paths}:
        resize_images(folder)
    descriptors = np.array([get_descriptor(p) for p in image_paths])
    nodes = [
        Node(path, desc, label=i)
        for i, (path, desc) in enumerate(zip(image_paths, descriptors))
    ]

    dists = cosine_distances(descriptors, descriptors)

    adj = {node: [] for node in nodes}
    for i, n1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            dist = dists[i, j]
            if dist < threshold:
                n2 = nodes[j]
                weight = 1 / max(dist, 1e-8) ** 2
                adj[n1].append((n2, weight))
                adj[n2].append((n1, weight))

    return nodes, adj

def connected_comps(adj, nodes):
    groups = {}
    for node in nodes:
        groups.setdefault(node.label, []).append(node)
    return list(groups.values())


def propagate_label(node, neighbors, adj):
    label_weights = {}
    for neighbor, weight in neighbors:
        label_weights[neighbor.label] = label_weights.get(neighbor.label, 0) + weight

    if label_weights:
        node.label = max(label_weights, key=label_weights.get)


def whispers(nodes, adj, iterations):
    component_counts = []
    for _ in range(iterations):
        node = random.choice(nodes)
        propagate_label(node, adj[node], adj)
        component_counts.append(len(connected_comps(adj, nodes)))
    return component_counts


def organize_photos(components, output_dir):
    for i, component in enumerate(components):
        subfolder = output_dir / f"component_{i}"
        subfolder.mkdir(parents=True, exist_ok=True)
        for node in component:
            # Copy or move the image to the subfolder
            # For example, using shutil.copy or shutil.move
            shutil.copy(node.image_path, subfolder / node.image_path.name)  # Replace with actual file operation

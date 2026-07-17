import random
import shutil
from collections import Counter
from pathlib import Path
from facenet_models import FacenetModel
import numpy as np
from skimage import io
from core.similarity import cosine_distances
from core.normalize import resize_images
detection_prob_threshold = 0.9
model = FacenetModel()
class Node:
    def __init__(self, image_path, descriptor, label):
        self.image_path = image_path
        self.descriptor = descriptor
        self.label = label

def get_descriptor(image_path):
    # Images come from phones/screenshots in mixed formats; some (e.g. HEIC)
    # skimage can't decode at all. Skip rather than crash the whole batch.
    try:
        image = io.imread(str(image_path))
    except Exception as e:
        print(f"Skipping {image_path}: could not read image ({e})")
        return None
    if image.shape[-1] == 4:
        image = image[..., :3]
    boxes, probabilities, landmarks = model.detect(image)
    if boxes is not None:
        mask = probabilities > detection_prob_threshold
        boxes = boxes[mask]
        probabilities = probabilities[mask]
    if boxes is None or len(boxes) == 0:
        print(f"Skipping {image_path}: no face detected")
        return None

    best = np.argmax(probabilities)
    descriptors = model.compute_descriptors(image, boxes)
    return descriptors[best]


def adj_list(image_paths, threshold):
    for folder in {p.parent for p in image_paths}:
        resize_images(folder)

    valid_paths, descriptors = [], []
    for p in image_paths:
        descriptor = get_descriptor(p)
        if descriptor is not None:
            valid_paths.append(p)
            descriptors.append(descriptor)
    descriptors = np.array(descriptors)
    nodes = [
        Node(path, desc, label=i)
        for i, (path, desc) in enumerate(zip(valid_paths, descriptors))
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
    # Patience scales with graph size. Each iteration only updates one random
    # node, so a small fixed window (the old `samecount > 10`) can look
    # "stable" by coincidence long before propagation has actually finished —
    # verified empirically: on a 28-node graph it stopped after 44 iterations
    # at 12 components, when true convergence (~6-7 components) took ~3000.
    patience = len(nodes) * 20
    stable_streak = 0
    for _ in range(iterations):
        node = random.choice(nodes)
        propagate_label(node, adj[node], adj)
        component_counts.append(len(connected_comps(adj, nodes)))
        if len(component_counts) > 1 and component_counts[-1] == component_counts[-2]:
            stable_streak += 1
        else:
            stable_streak = 0
        if stable_streak > patience:
            break
    return component_counts

def organize_photos(components, output_dir=Path("result")):
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names = set()
    for i, component in enumerate(components):
        names = [node.image_path.parent.name for node in component]
        person_name = Counter(names).most_common(1)[0][0]
        if person_name in used_names:
            person_name = f"{person_name}_{i}"
        used_names.add(person_name)

        subfolder = output_dir / person_name
        subfolder.mkdir(parents=True, exist_ok=True)
        for node in component:
            shutil.copy(node.image_path, subfolder / node.image_path.name)

if __name__ == "__main__":
    input_dir = Path("known_faces")
    output_dir = Path("result")

    # known_faces/ has per-person subfolders, and images come from phones/
    # screenshots in mixed formats/cases, so glob recursively over extensions.
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    image_paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Cosine-distance threshold (not detection probability) — see README.
    threshold = 0.2846

    nodes, adj = adj_list(image_paths, threshold)
    print(f"Built graph: {len(nodes)} nodes")

    iterations = 200 * len(nodes)
    component_counts = whispers(nodes, adj, iterations)

    import matplotlib
    matplotlib.use("Agg")  # headless-safe; we only save the figure, never show()
    import matplotlib.pyplot as plt
    plt.plot(component_counts)
    plt.xlabel("iteration")
    plt.ylabel("number of connected components")
    plt.title("Whispers convergence")
    plt.savefig("result_convergence.png")
    print("Saved convergence plot to result_convergence.png")

    components = connected_comps(adj, nodes)
    print(f"Converged to {len(components)} components")

    organize_photos(components, output_dir)
    print(f"Organized photos into {output_dir}/")
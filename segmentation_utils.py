import numpy as np

from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2


def get_clusters_from_grayscale_image(image):

    populated_pixels = get_image_populated_pixels(image)

    if len(populated_pixels) < 20:
        return [], [], []

    # Apply DBSCAN
    dbscan = DBSCAN(eps=8, min_samples=10)  # Adjust parameters as needed
    labels = dbscan.fit_predict(populated_pixels)

    # Group pixels by cluster label
    clusters = defaultdict(list)
    for pixel, label in zip(populated_pixels, labels):
        if label != -1:  # Skip noise
            clusters[label].append(pixel)

    # Convert to list of numpy arrays (optional, for convenience)
    grouped_clusters = [np.array(cluster) for cluster in clusters.values()]

    return grouped_clusters


def get_image_populated_pixels(image):
    # Find the indices where the pixel value is greater than 100
    populated_indices = np.argwhere(image > 100)
    return populated_indices


def draw_detection_bound_boxes(image, detections_data):
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for detection_data in detections_data:
        box_x, box_y, box_w, box_h = detection_data["coords_data"]
        color = detection_data["color"]
        confidence = detection_data["confidence"]
        label = f"{detection_data['label']}: {str(round(confidence, 2))}"

        cv2.rectangle(
            image,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            color,
            2,
        )
        cv2.putText(
            image, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return image

import numpy as np

from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2

# from colors_utils import color_dict
# from collections import defaultdict
# import matplotlib.pyplot as plt

# FRAMES_LIMIT = 1000
# COLORS_LIST = list(color_dict.values())


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


# def resize_image(image, new_width, new_height):
#     return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# def crop_image(image, x_min, x_max, y_min, y_max):
#     return image[y_min:y_max, x_min:x_max]


# def threshold_image(image, threshold=128):
#     """
#     Transforms an image: if a pixel value is above the threshold,
#     it turns white (255), otherwise, it turns black (0).

#     Args:
#         image_path (str): Path to the input image.
#         threshold (int): Threshold value (0-255).

#     Returns:
#         numpy.ndarray: The thresholded binary image.
#     """
#     # Convert to grayscale if the image is colored
#     if len(image.shape) == 3:  # Check if image has multiple channels
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply thresholding
#     _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

#     return binary_image


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

# def plot_array(y, x_lim = None, y_lim = None):
#     # x-axis: progressive integers starting from 0
#     x = np.arange(len(y))

#     # Plot
#     plt.plot(x, y, marker='o', linestyle='-', color='g')  # 'o' for dots, '-' for lines

#     if x_lim is not None:
#         plt.xlim(x_lim[0], x_lim[1])
#     if y_lim is not None:
#         plt.ylim(y_lim[0], y_lim[1])        
#     plt.grid(True)  # Enable grid
#     plt.xlabel('$\Theta_d$')
#     plt.ylabel('$\\rho$')
#     # plt.title('Plot of Array Values')
#     plt.title('Relative distance vs discrete angle')
#     plt.show()
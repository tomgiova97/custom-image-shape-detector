import cv2
import numpy as np

class MultiShapeDetector:
    def __init__(self, shape_images, detect_params):

        self.shape_images = shape_images
        self.detect_params = detect_params
        self.shapes_segm_pixel_list = self.get_shapes_segm_pixel_list()
        self.angle_res = np.pi / 24
        self.shapes_contour_functions = self.get_shapes_contour_functions()
        self.box_color = [0, 255, 0]  # Green in Opencv BGR

    def compare(self, segm_pixel_list):
        box_data = {"color": self.box_color}

        contour_function, coords_data = self.get_contour_function_from_segm_pixel_list(
            segm_pixel_list
        )

        overlaps = self.calc_contour_overlapping_with_shapes(contour_function)
        detection = self.find_object_detection_from_overlaps(overlaps)

        if detection[0] != -1:
            shape_detect_params = self.detect_params[detection[0]]

            box_data["coords_data"] = coords_data
            box_data["label"] = shape_detect_params["label"]
            box_data["confidence"] = detection[1]

            return box_data
        else:
            return {}

    def get_shapes_segm_pixel_list(self):
        shapes_segm_pixel_list = []
        for shape_image in self.shape_images:
            shapes_segm_pixel_list.append(
                self.get_segm_pixel_list_from_image(shape_image)
            )

        return shapes_segm_pixel_list

    def get_segm_pixel_list_from_image(self, image):
        segm_pixel_list = np.argwhere(image > 0)
        return segm_pixel_list

    def get_shapes_contour_functions(self):
        shapes_contour_functions = []
        for segm_pixel_list in self.shapes_segm_pixel_list:
            shape_contour_function, _ = self.get_contour_function_from_segm_pixel_list(
                segm_pixel_list
            )
            shapes_contour_functions.append(shape_contour_function)
        return shapes_contour_functions

    def get_contour_function_from_segm_pixel_list(self, segm_pixel_list):
        # Step 1: Initialize
        angle_bins = int(np.round(2 * np.pi / self.angle_res))
        contour_function = np.zeros(angle_bins)

        # Step 2: Get center of mass
        x_cm, y_cm = self.calc_segm_pixel_list_center_of_mass_coords(segm_pixel_list)

        # Step 3: Extract x and y
        y_segm = segm_pixel_list[:, 0]
        x_segm = segm_pixel_list[:, 1]

        y_min, y_max = y_segm.min(), y_segm.max()
        x_min, x_max = x_segm.min(), x_segm.max()

        # Step 4: Compute distances and angles
        dy = y_segm - y_cm
        dx = x_segm - x_cm
        angles = np.arctan2(-dy, dx) % (2 * np.pi)  # Ensure angles in [0, 2π)
        dists = np.sqrt(dx**2 + dy**2)
        norm_dists = dists / ((y_max - y_min) / 2 + 1e-8)  # Normalize

        # Step 5: Bin the angles
        bin_indices = np.floor(angles / self.angle_res).astype(int)

        # Step 6: Assign max distance per bin
        for i in range(angle_bins):
            # mask is an array with len of segm_pixel_list
            # that is 1 for the pixels who fall in bin i and 0 otherwise
            mask = bin_indices == i
            if np.any(mask):
                contour_function[i] = np.max(norm_dists[mask])

        # Step 7: Fill gaps in the contour function
        if np.any(contour_function == 0):
            # Get indices of non-zero and zero bins
            non_zero_indices = np.where(contour_function != 0)[0]
            zero_indices = np.where(contour_function == 0)[0]

            for i in zero_indices:
                # Find closest non-zero index (circularly)
                distances = np.abs(non_zero_indices - i)
                circular_distances = np.minimum(distances, angle_bins - distances)
                nearest_idx = non_zero_indices[np.argmin(circular_distances)]
                contour_function[i] = contour_function[nearest_idx]

        return contour_function, (x_min, y_min, x_max - x_min, y_max - y_min)

    def calc_segm_pixel_list_center_of_mass_coords(self, segm_pixel_list):
        y_cm, x_cm = np.mean(segm_pixel_list, axis=0)
        return (x_cm, y_cm)

    def calc_contour_overlapping_with_shapes(self, contour_function):
        overlaps = []
        for shape_contour_function in self.shapes_contour_functions:
            overlaps.append(
                self.calc_vectors_overlapping(shape_contour_function, contour_function)
            )
        return overlaps

    def calc_vectors_overlapping(self, vect_1, vect_2):
        diff = np.abs(vect_1 - vect_2)
        similarity = 1 - diff

        # Clip to avoid negative values just in case
        similarity = np.clip(similarity, 0, 1)

        return np.mean(similarity)

    def find_object_detection_from_overlaps(self, overlaps):
        if len(overlaps) != len(self.detect_params):
            raise Exception(
                f"Overlaps must have the same shape of the input detection params"
            )
        detection = (-1, 0)
        for i in range(0, len(overlaps)):
            shape_detect_params = self.detect_params[i]
            shape_sigm_params = shape_detect_params["sigm_params"]
            confidence = self.sharp_sigmoid_func(
                overlaps[i],
                shape_sigm_params[0],
                shape_sigm_params[1],
                # shape_sigm_params[2],
            )
            if (
                confidence >= shape_detect_params["detect_thresh"]
                and confidence > detection[1]
            ):
                detection = (i, confidence)
        return detection
    
    def sharp_sigmoid_func(self, x, x_0=0.5, sharpness = 10):
        if x >= x_0:
            return 1 +( x_0 - 1)*np.exp(- sharpness*(x - x_0))
        else:
            return  x_0*np.exp(sharpness*(x - x_0))    
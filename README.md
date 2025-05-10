# Multi-Shape Custom Detector

This repository provides an implementation example of the shape detection algorithm described in [insert paper link here].  
For a complete explanation of the algorithm, please refer to the accompanying article.

An example of usage can be found in the notebook file: [`main.ipynb`](main.ipynb).

---

## Dependencies
To run this Python project the required dependencies are Numpy and OpenCV

## Initializing the Shape Detector

Similar to other deep learning-based object detectors like YOLO, this algorithm can detect multiple shapes simultaneously.

Suppose we want to detect both *persons* and *dogs* in an image. For each target shape, the detector must be initialized with:

1. A **shape image**
2. A **set of detection parameters**

The shape image should be a grayscale image where:
- Pixels belonging to the shape have a value of **255**
- Background pixels are **0**

Detection parameters must be provided as a dictionary in the following format:

    {
        "label": "person",
        "threshold": 0.75,
        "sigm_params": (0.7, 10)
    }

- label: A string identifier for the shape

- threshold: The similarity threshold required for a detection

- sigm_params: A tuple representing the parameters (center, steepness) of the modified sigmoid function, as described in the paper



## Extract segmentation pixels coordinates
The detector operates on a list of pixel coordinates representing object regions in the image.

To extract these regions, we use the DBSCAN clustering algorithm to identify distinct groups of foreground pixels.
This allows the detector to separate and analyze individual objects in the image.


## Performing Detection
After initializing the detector and extracting object regions, we can pass the segmented clusters to the detector using the compare method.

This method compares each cluster against all registered shape templates.
If the similarity with any shape exceeds its corresponding threshold, a detection is returned in the following format:

    {
        "label": "person",
        "confidence": 0.95,
        "coords_data" : (x_min, y_min, width, height)
        "color": [0, 255, 0]
    }

label: The class of the most similar shape

confidence: The computed similarity score

coords_data: Bounding box coordinates of the detected object

color: (Optional) Color used for drawing the bounding box

If no shape exceeds the similarity threshold, the method returns an empty result:


    {}


## Contact
For questions or feedback, feel free to open an issue or reach out at [tg030397@gmail.com]


## License
This software is free to use under MIT License. If using this software for a public project, please cite the article ().


MIT License

Copyright (c) [2025] [Tommaso Giovannelli]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
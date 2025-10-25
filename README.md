# Computer Vision Assignment 2 by st126488: Classification, Detection, and Tracking

 ⚠️ Important Note on Datasets

Please note that the datasets used for this project are **not included** in this zip file due to their large file size.

* **Task 1 (Classification):** The `task_1_classification.ipynb` script is configured to automatically download the **CIFAR-10** dataset upon first run.
* **Task 2 (Detection):** You will need to provide your own test images and a test video. The script was run using the **COCO128** dataset for images and a file named `test.gif` for video.
* **Task 3 (Tracking):** You will need to provide your own test video. The script was run using a file named `test.gif`.

## Setup and Dependencies

To run these notebooks, you will need a Python environment with the following core libraries installed:

* `torch` & `torchvision`
* `ultralytics` (for YOLOv8)
* `opencv-python`
* `opencv-contrib-python` (required for the trackers in Task 3)
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`

It is highly recommended to run this code in an environment with a **CUDA-compatible GPU**, as the models in Task 1 and Task 2 are computationally intensive.

---

## Task 1: Image Classification

**Notebook:** `st126488_notebook_task_{1}.ipynb`

### What I Did
This task compares the performance of different models and optimizers for image classification on the CIFAR-10 dataset.

* **Models:**
    * `SimpleCNN`: A basic, shallow convolutional neural network built from scratch.
    * `ResNet-18`: A deep, pre-trained model using transfer learning.
* **Optimizers:**
    * `SGD`: Stochastic Gradient Descent with momentum.
    * `Adam`: The Adam optimizer.

The notebook trains all 4 combinations (2 models x 2 optimizers) for 10 epochs and generates loss/accuracy plots and confusion matrices for each run.

### Key Findings
* **Accuracy:** The pre-trained **ResNet-18** model (~78-81% validation accuracy) was significantly more accurate than the `SimpleCNN` (~62-64% validation accuracy).
* **Speed:** The `SimpleCNN` was much faster to train (~150 seconds) compared to `ResNet-18` (~1100-1200 seconds), which is expected due to its depth and the larger image size required (224x224 vs 32x32).
* **Optimizer:** For both models, **Adam** provided a slight performance edge over SGD in this 10-epoch test, achieving the highest overall accuracy of **80.95%** (ResNet-18 + Adam).

---

## Task 2: Object Detection

**Notebook:** `st126488_notebook_task_{2}.ipynb`

### What I Did
This task benchmarks a two-stage object detector against a single-stage object detector to compare their performance, with a focus on speed.

* **Models:**
    * `Faster R-CNN`: A two-stage detector (from `torchvision`).
    * `YOLOv8n`: A single-stage, real-time detector (from `ultralytics`).
* **Metrics:**
    * **Average FPS** on a test video.
    * **Inference Time (ms)** on a set of test images.
    * **Model Size** on disk.

### Key Findings
* **Speed (Video FPS):** **YOLOv8n was dramatically faster** (~43 FPS) and suitable for real-time applications. Faster R-CNN was very slow (~6 FPS).
* **Speed (Image Inference):** YOLOv8n was over 4.5 times faster (~41 ms per image) than Faster R-CNN (~186 ms per image).
* **Model Size:** YOLOv8n is extremely lightweight (**~6.25 MB**) compared to Faster R-CNN (**~163 MB**).
* **Conclusion:** For applications where speed is critical, YOLOv8n is the clear winner. Faster R-CNN may provide higher accuracy (not measured here) but at a significant computational cost.

---

## Task 3: Object Tracking

**Notebook:** `st126488_notebook_task_{3}.ipynb`

### What I Did
This task compares the speed and robustness of several classical object trackers available in OpenCV.

* **Trackers:**
    * `KCF` (Kernelized Correlation Filters)
    * `CSRT` (Discriminative Correlation Filter with Channel and Spatial Reliability)
    * `MOSSE` (Minimum Output Sum of Squared Error)
    * `MIL` (Multiple Instance Learning)
    * `TLD` (Tracking-Learning-Detection)
* **Process:** The script first prompts the user to select an object by drawing a bounding box on the first frame of a video. It then runs each tracker sequentially on the full video, starting with the same initial box, and reports its average FPS and success rate.

### Key Findings
* **Accuracy:** On the simple test video, **all trackers achieved a 100% success rate**, successfully tracking the object without failure.
* **Speed (FPS):** There was a massive difference in performance:
    * **Fastest:** `MOSSE` was the clear winner, achieving an extremely high **~837 FPS**.
    * **Fast:** `KCF` (~157 FPS) and `TLD` (~86 FPS) were also very fast.
    * **Slowest:** `CSRT` (~27 FPS) and `MIL` (~24 FPS) were significantly slower, though still near real-time.
* **Conclusion:** For this specific tracking task, `MOSSE` offered the best performance by a wide margin, though `KCF` and `TLD` are also excellent high-speed options. `CSRT` is often more robust on challenging videos, which explains its slower speed.

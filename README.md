# <p align="center">🎯 Pose Estimation in High Vibrational Systems: Classical vs Deep Learning Approaches</p>

## 🌐 Project Overview

Pose estimation in dynamic, high-vibrational environments presents a unique challenge, particularly in military applications where real-time object tracking and situational awareness are crucial. High-motion and vibration often degrade traditional methods, making innovative solutions necessary. This study compares two approaches: a classical computer vision technique, which is fast but limited in accuracy, and a deep learning-based strategy like YOLO v11m, which excels in accuracy but requires high computational power. By exploring both methods, we aim to improve pose estimation for military operations, surveillance, and autonomous systems in complex environments.


## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pose-estimation-project

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
python yolo11.py
```

## 🔬 Methodological Approaches

### 🖥️ Phase 1: Classical Computer Vision Approach

#### Performance Metrics
- **Frame Rate**: Up to 200 FPS
- **Accuracy**: 85%
- **Computational Complexity**: Low
- **Training Data**: Zero

#### Technical Arsenal
- **Key Techniques**:
  - Gaussian Blur
  - Canny Edge Detection
  - Contour Analysis
  - Dynamic Shape Scoring <br>
#### Detection Strategy
1. **Multi-Parameter Object Recognition**
   - Sophisticated contour approximation
   - Shape circularity calculation
   - Temporal consistency scoring
   - Color-based feature extraction

2. **Advanced Visual Augmentation**
   - Real-time motion trail visualization
   - Energy field particle simulation
   - Dynamic 3D wireframe rendering
   - Adaptive color tracking <br><br>
<img src="https://github.com/tejas2510/Pose-Estimation-in-high-vibrational-systems/blob/master/assets/steps.gif?raw=true" alt="Classical Approach">

<img src="https://github.com/tejas2510/Pose-Estimation-in-high-vibrational-systems/blob/master/assets/classical.gif?raw=true" alt="Classical Approach">

### 📊 Output Metric Plots

<img src="https://github.com/tejas2510/Pose-Estimation-in-High-Vibrational-Systems/blob/master/outputs/metric_plots_20241129230031.png?raw=true" alt="plots" height="500px" width="750px">

### 🤖 Phase 2: Deep Learning Approach (YOLO v11m)

#### Performance Metrics
- **Frame Rate**: 25 FPS
- **Accuracy**: 99%
- **Computational Complexity**: High
- **Training Data**: 3000+ custom images

#### Technical Arsenal
- **Core Technologies**:
  - PyTorch
  - YOLO v11m Architecture
  - Advanced Neural Networks

#### Sophisticated Detection Methodology
1. **Comprehensive Training Pipeline**
   - Custom dataset of 3000+ annotated images
   - Rigorous data augmentation
   - Cross-validation strategies
   - Hyperparameter optimization

2. **Advanced Neural Network Features**
   - Multi-scale feature extraction
   - Robust object localization
   - Depth prediction integration
   - Real-time inference capabilities <br> <br>
<img src="https://github.com/tejas2510/Pose-Estimation-in-high-vibrational-systems/blob/master/assets/dl_approach.gif?raw=true" alt="DL Approach">

## 📊 Comparative Analysis

| Characteristic | Classical Approach | Deep Learning Approach |
|---------------|--------------------|-----------------------|
| **Frame Rate** | 200 FPS | 25 FPS |
| **Accuracy** | 85% | 99% |
| **Computational Resources** | Low | High |
| **Training Requirement** | None | Extensive |
| **Adaptability** | Limited | High |

## 🔍 Key Insights: Classical vs Deep Learning Approaches

- **Classical Approach**:
  - **Strengths**:
    - **Frame Rate**: Up to 200 FPS with low computational requirements.
    - **Training Data**: None required (zero training).
    - **Computational Requirements**: Very Low, doesn't need any expensive hardware.

  - **Weaknesses**:
    - **Accuracy**: 85%, which may struggle in complex or dynamic environments.
    - **Adaptability**: Limited to predefined techniques and cannot learn from new data.
  - **Applications**: Robotics, Surveillance, Augmented Reality, Industrial Automation, Military Applications (e.g., real-time object recognition, situational awareness in field operations).

- **Deep Learning Approach (YOLO v11m)**:
  - **Strengths**:
    - **Accuracy**: 99%, ideal for precise pose estimation in challenging environments.
    - **Training Data**: Requires 3000+ custom images for training.
  - **Weaknesses**:
    - **Frame Rate**: 25 FPS, slower than the classical approach.
    - **Computational Requirements**: High, demanding powerful hardware for training and inference.
  - **Applications**: Autonomous Vehicles, Medical Imaging, Advanced Robotics, Sports Analytics.

## 🗺️ Depth Estimation
- Integrated MiDaS depth estimation
- Real-time depth map generation
- Multi-resolution depth prediction

## 📝 Future Work
- Develop hybrid detection algorithms
- Reduce computational overhead
- Improve real-time performance
- Expand dataset diversity

## 🎓 Credits

- **MiDaS** (Depth Estimation):  
  MiDaS is used for depth estimation, providing high-quality depth maps in complex scenes.  
  Paper: [MiDaS: A Multi-Scale Network for Depth Estimation](https://arxiv.org/abs/1907.01341)

- **YOLO 11m** (Object Detection):  
  YOLO 11m is the object detection model employed in this study for precise pose estimation & object detection.  
  Paper: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/html/2410.17725v1)

We would like to thank the respective authors and communities for their contributions to the open-source ecosystem!


## ✨ Contributors

Contributions are always welcome! Please check our [Contributing Guidelines](/CONTRIBUTING.md)

<a href="https://github.com/tejas2510/Pose-Estimation-in-high-vibrational-systems/graphs/contributors">
<a href="https://github.com/tejas2510/"><img src="https://github.com/tejas2510.png" width="60" height="60" style="border-radius: 50%; object-fit: cover; margin-right: 10px;" /></a>
<a href="https://github.com/ShashankBhat-18/"><img src="https://github.com/ShashankBhat-18.png" width="60" height="60" style="border-radius: 50%; object-fit: cover;" /></a>
</a>

## 📧 Contact

For more information, collaborations, or inquiries:
- Email: [for.tejaspatil@gmail.com](mailto:for.tejaspatil@gmail.com)
- Discord: #klayjensen
- Project Link: [GitHub - Precision Pose Estimation](https://github.com/tejas2510/Pose-Estimation-in-high-vibrational-systems)

**Made with ❤️ & Python**

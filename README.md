# 🧬 EthoGrid_Nereis  
## High-Throughput Tracking, Pose Estimation, Segmentation, and Behavioral Analysis of *Nereis*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![UI](https://img.shields.io/badge/UI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![Deep Learning](https://img.shields.io/badge/AI-YOLO%20%7C%20Pose%20Estimation-purple.svg)](#)

---

## 📌 Overview

**EthoGrid_Nereis** is a dedicated, research-grade desktop application for **automated behavioral analysis of *Nereis*** using video recordings.  
The platform supports **object detection, instance segmentation, pose estimation, and multi-arena tracking** within a unified and transparent workflow.

The system is specifically designed for **high-throughput screening experiments**, enabling large-scale, reproducible behavioral phenotyping of *Nereis* under diverse experimental conditions.

> ⚠️ This application is optimized for *Nereis* morphology and locomotion dynamics and is **not a generic worm-tracking tool**.

---

<p align="center">
  <img 
    src="https://github.com/yousaf2018/EthoGrid_Nereis/blob/main/images/logo.png"
    width="200"
    height="200"
    style="border-radius: 50%; object-fit: cover;"
  >
</p>



<p align="center">
  <em>EthoGrid_Nereis an end-to-end AI-powered ethomics platform for marine annelid research</em>
</p>

---

## 🎯 Key Features

### 🧠 AI-Based Behavioral Intelligence
- **Detection** of *Nereis* individuals using YOLO-based deep learning models
- **Instance Segmentation** for pixel-accurate body contour extraction
- **Pose Estimation** for fine-scale posture and bending dynamics analysis
- Support for multiple inference modes within a single application

### ⚡ High-Throughput Screening
- Batch processing of large video datasets
- GPU-accelerated inference with CPU fallback
- Multi-arena and multi-tank experimental support
- Designed for toxicology, neuroethology, and pharmacological studies

### 🧩 Interactive Grid-Based Arena Mapping
- Flexible virtual grid system aligned to experimental layouts
- Interactive translation, rotation, and scaling
- Persistent grid configuration for reproducibility
- Automated arena assignment during batch processing

### 🧹 Data Cleaning & Quality Control
- Confidence-based detection filtering
- Removal of duplicated or spurious detections
- Robust handling of missing detections
- Transparent CSV-based intermediate data outputs

### 📊 Behavioral Endpoints & Statistics
- Trajectory- and zone-based behavioral metrics
- Pose-derived descriptors
- Automated statistical testing and visualization
- Export of publication-ready figures and tables

---

## 🔁 Complete Workflow

1. Video preprocessing and optional segmentation  
2. AI inference (detection / segmentation / pose estimation)  
3. Grid alignment and arena annotation  
4. Batch processing for high-throughput datasets  
5. Endpoint extraction  
6. Statistical analysis and visualization  

---

## 📂 Output Files

- Annotated videos (`.mp4`)
- Raw and cleaned tracking data (`.csv`)
- Excel summaries (`.xlsx`)
- Trajectory plots and heatmaps (`.png`)
- Statistical result reports

---

## 🛠 Installation

```bash
git clone https://github.com/yousaf2018/EthoGrid.git
cd EthoGrid

conda create -n ethogrid-nereis python=3.8 -y
conda activate ethogrid-nereis

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python main.py
```

---

## 📚 Documentation

- `DEVELOPER_GUIDE.md` – Code architecture and implementation details  
- `STATISTICAL_ANALYSIS_GUIDE.md` – Statistical methods and endpoint definitions  

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository  
2. Create a feature branch  
3. Commit your changes  
4. Push to your branch  
5. Open a pull request  

---

## 🙏 Acknowledgements

This application was developed in the **[Laboratory of Professor Chung-Der Hsiao](https://cdhsiao.weebly.com/pi-cv.html)** in collaboration with **Chung Yuan Christian University, Taiwan 🇹🇼**.  
Special credit and sincere gratitude are extended to **Professor Hsiao**, who shared his extensive research experience in biology and multiple domains, providing invaluable guidance and supervision throughout the development of this application.

<p align="center">
  <a href="https://www.cycu.edu.tw/">
    <img src="https://raw.githubusercontent.com/yousaf2018/EthoGrid/main/images/cycu.jpg" width="250">
  </a>
</p>

---

## 📜 License

Distributed under the **MIT License**.  
See the `LICENSE` file for details.

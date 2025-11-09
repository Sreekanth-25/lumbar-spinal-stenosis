
## 🩻 Overview  

This project automates the **detection, quantification, and reporting** of Lumbar Spinal Stenosis (LSS) from MRI scans.  
It integrates medical image segmentation, morphological analysis, and deep classification into one explainable system.

### 🔬 Core Workflow
1. **Segmentation:** ResNet50-UNet extracts spinal canal and stenotic regions.  
2. **Feature Extraction:** Computes morphological parameters from masks (area, compactness, opening ratio).  
3. **Rule-Based Grading:** Quantifies stenosis severity (Normal → Severe).  
4. **Deep Classification:** DenseNet201 classifies MRI images directly by severity.  
5. **Report Generation:** Produces a `.docx` clinician-style diagnostic report with images, metrics, and summaries.

---

## ⚙️ Installation  

Run in **Kaggle Notebook** or locally with GPU support.

### Dependencies  
```bash
pip install torch torchvision opencv-python matplotlib scikit-learn pillow python-docx

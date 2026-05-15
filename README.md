# A-Computer-Vision-Approach-for-Detecting-Real-vs-AI-Generated-Images-2026-
Deep learning framework for AI-generated image detection using spatial and frequency-domain analysis.
# Versus Fake: Multimodal AI-Generated Image Detection

A deep learning framework for detecting AI-generated images using multimodal CLIP architectures, forensic frequency analysis, and transformer-based representation learning.

---

# Overview

The rapid advancement of generative AI models has significantly increased the realism of synthetic imagery, creating challenges in digital media authenticity and visual forensics. This project investigates multiple deep learning approaches for detecting AI-generated images through:

* Image-only CLIP classification
* Multimodal image-text fusion
* Frequency-domain forensic preprocessing
* Fine-tuned transformer-based architectures

The framework evaluates how semantic information, visual representations, and spectral forensic artifacts contribute to real-vs-fake image detection.

---

# Objectives

This project aims to:

* Detect AI-generated images accurately
* Compare image-only and multimodal approaches
* Explore frequency-domain forensic preprocessing
* Analyze transformer-based visual representations
* Evaluate binary and multi-class classification performance
* Build a reproducible research-oriented pipeline

---

# Experiments

Three major experiments were conducted.

---

# Experiment 1 — Image-Only CLIP

A frozen CLIP visual encoder is used to extract image embeddings for classification.

### Key Characteristics

* Image-only pipeline
* Frozen CLIP backbone
* Visual embedding classification
* Data augmentation
* Lightweight classifier head

### Results

| Metric             | Score  |
| ------------------ | ------ |
| Task B Accuracy    | 76.80% |
| Task B Weighted F1 | 0.764  |
| Task A Accuracy    | 94.79% |
| Task A Weighted F1 | 0.948  |

### Scientific Observation

The pretrained CLIP visual encoder provides strong semantic and structural representations, allowing high binary fake detection performance even without textual information.

---

# Experiment 2 — Multimodal CLIP (Image + Caption)

A multimodal architecture combining:

* image embeddings
* caption embeddings

through feature fusion.

### Key Characteristics

* Image + text fusion
* Frozen CLIP architecture
* Multimodal representation learning
* Feature concatenation
* Transformer-based embeddings

### Results

| Metric             | Score  |
| ------------------ | ------ |
| Task B Accuracy    | 77.67% |
| Task B Weighted F1 | 0.772  |
| Task A Accuracy    | 94.64% |
| Task A Weighted F1 | 0.947  |

### Scientific Observation

Captions provide small contextual improvements, but visual information remains the dominant factor for synthetic image detection.

---

# Experiment 3 — Frequency-Domain Forensic CLIP

A forensic preprocessing pipeline was introduced using:

* FFT
* autocorrelation
* residual extraction

before multimodal CLIP classification.

### Key Characteristics

* Frequency-domain preprocessing
* Spectral artifact extraction
* Autocorrelation analysis
* Fine-tuned visual encoder
* Multimodal fusion

### Results

| Metric             | Score  |
| ------------------ | ------ |
| Task B Accuracy    | 68.80% |
| Task B Weighted F1 | 0.686  |
| Task A Accuracy    | 88.95% |
| Task A Weighted F1 | 0.897  |

### Scientific Observation

Frequency-domain forensic features contain strong fake-vs-real information but reduce semantic visual information, resulting in lower multi-class classification performance.

---

# Final Comparative Analysis

| Experiment              | Task B Accuracy | Task A Accuracy |
| ----------------------- | --------------- | --------------- |
| Image-Only CLIP         | 76.80%          | 94.79%          |
| Multimodal CLIP         | **77.67%**      | 94.64%          |
| Frequency Forensic CLIP | 68.80%          | 88.95%          |

---

# Key Findings

### 1. Visual Features Are Most Important

CLIP visual embeddings provided the strongest contribution to classification performance.

### 2. Captions Offer Limited Improvement

Textual information improved performance slightly but did not dominate classification decisions.

### 3. Frequency Artifacts Help Binary Detection

Spectral forensic preprocessing improved fake-vs-real signal extraction but reduced semantic representation quality.

### 4. CLIP Handles Semantic Structure Extremely Well

Even frozen CLIP embeddings achieved strong classification performance.

---

# Methodology

---

# 1. Frequency-Domain Forensic Preprocessing

The forensic pipeline includes:

1. RGB to grayscale conversion
2. Gaussian denoising
3. Residual extraction
4. Fast Fourier Transform (FFT)
5. Power spectrum calculation
6. Autocorrelation mapping
7. Log normalization
8. RGB conversion

This process exposes:

* repetitive synthesis patterns
* hidden spectral inconsistencies
* artificial generation artifacts

commonly present in AI-generated images.

---

# 2. Multimodal Fusion

Image embeddings and text embeddings are concatenated:

```python id="vgnuc9"
fusion = torch.cat([image_feat, text_feat], dim=1)
```

The fused representation is passed through:

* fully connected layers
* layer normalization
* GELU activation
* dropout regularization

before final classification.

---

# 3. CLIP Backbone

The project uses:

```text id="w2dwo5"
openai/clip-vit-base-patch32
```

CLIP provides:

* transformer-based visual understanding
* joint image-text representation learning
* strong semantic feature extraction

---

# Technologies Used

| Technology                | Purpose                   |
| ------------------------- | ------------------------- |
| Python                    | Core programming language |
| PyTorch                   | Deep learning framework   |
| Hugging Face Transformers | CLIP implementation       |
| OpenCV                    | Image preprocessing       |
| NumPy                     | Numerical computation     |
| Scikit-learn              | Evaluation metrics        |
| Matplotlib                | Visualization             |
| Seaborn                   | Statistical plotting      |


# Requirements

```text id="m8uh6t"
torch
torchvision
transformers
datasets
numpy
opencv-python
scikit-learn
matplotlib
seaborn
tqdm
Pillow
```

---

# Dataset

This project uses:

```text id="p4w7sn"
Rajarshi-Roy-research/Defactify_Image_Dataset
```

loaded directly through the Hugging Face `datasets` library.

---

# Reproducibility

To ensure reproducibility:

* Global seeds are fixed
* Deterministic CUDA behavior is enabled
* DataLoader workers are seeded consistently

---

# Model Architecture

### Components

* CLIP visual encoder
* CLIP text encoder
* Multimodal fusion network
* Fully connected classifier

### Fusion Layers

```text id="drzc7d"
Linear → LayerNorm → GELU → Dropout
```

---

# Challenges Encountered

* Domain shift between RGB images and frequency-domain representations
* Fine-grained multi-class classification difficulty
* Balancing semantic and forensic feature extraction

---

# Future Work

Potential future improvements include:

* Vision Transformer enhancements
* Explainable AI visualization
* Diffusion-model-specific forensics
* Video deepfake detection
* Adaptive multimodal fusion strategies

---

# Conclusion

This project demonstrates that:

* transformer-based visual representations are highly effective for AI-generated image detection,
* multimodal fusion provides small but meaningful improvements,
* frequency-domain forensic analysis contains valuable fake-detection information,
* and combining semantic and forensic representations remains a promising research direction.

---

# License

This project is licensed under the MIT License.

---

# Author

**Sajedah Abdulelah Alqudaihi**
**Hanan Saeed Alshumrani**
**Leena Nabil Bukair**
**Layan Mohsen Alduais**



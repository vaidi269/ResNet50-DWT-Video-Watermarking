# ResNet50-DWT-Video-Watermarking
ResNet50-assisted frame selection with robust DWT-based RGB video watermarking, watermark extraction, NCC verification, and 3D video cube visualization for reproducible visual computing research.
# ResNet50-Assisted Frame Selection for Robust DWT-Based Color Video Watermarking

This repository contains the official implementation associated with our manuscript submitted to **The Visual Computer**.

## Overview

This work proposes a robust video watermarking framework that combines:

* ResNet50-based semantic frame selection
* Discrete Wavelet Transform (DWT)
* RGB watermark embedding
* Robust watermark extraction and verification

Unlike conventional watermarking approaches, the proposed method uses deep feature representations extracted using ResNet50 to intelligently select semantically rich video frames for watermark embedding, thereby improving robustness while maintaining high visual quality.

---

## Methodology

### Embedding Pipeline

Video Frames
↓
ResNet50 Feature Extraction
↓
Frame Importance Scoring
↓
Top Frame Selection
↓
DWT-Based RGB Watermark Embedding
↓
Inverse DWT
↓
Watermarked Video Generation

### Extraction Pipeline

Watermarked Video
↓
Selected Frame Verification
↓
DWT-Based Watermark Extraction
↓
RGB Watermark Reconstruction
↓
NCC Verification

---

## Features

* Deep-learning-assisted frame selection
* RGB watermark embedding
* High PSNR and SSIM preservation
* Robust watermark extraction
* NCC-based verification
* 3D video cube visualization

---

## Dependencies

Install required libraries using:

```bash
pip install -r requirements.txt
```

---

## Running the Code

### Watermark Embedding

```bash
python embedding.py
```

### Watermark Extraction

```bash
python extraction.py
```

---

## Outputs

The framework generates:

* Watermarked video
* Extracted watermark images
* NCC/PSNR/SSIM plots
* Selected frame indices
* 3D video cube visualization

<img width="509" height="421" alt="WhatsApp Image 2025-10-17 at 15 34 48_756d66ca" src="https://github.com/user-attachments/assets/b717959f-636d-4a72-bcab-3d1b902d85d7" />

---

## Repository Contents

| File                       | Description                     |
| -------------------------- | ------------------------------- |
| embedding.py               | Watermark embedding pipeline    |
| extraction.py              | Watermark extraction pipeline   |
| selected_frames.npy        | ResNet50-selected frame indices |
| watermarked_video_cube.npy | 3D watermarked video cube       |
| requirements.txt           | Required dependencies           |

---

## Citation Notice

This repository is directly related to the manuscript currently submitted to **The Visual Computer**.

If you use this code, dataset, or methodology in your research, please cite the corresponding manuscript.

---

## DOI and Permanent Archive

A permanent archived version of this repository with DOI is available through Zenodo.

(Insert Zenodo DOI here after publication)

---

## License

This project is released for academic and research purposes.

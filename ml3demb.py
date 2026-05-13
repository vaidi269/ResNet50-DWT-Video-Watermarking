# =============================================================
# FULL EMBEDDING CODE
# ResNet50-Based Frame Selection + Classical DWT Watermarking
# =============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import pywt
import pyvista as pv
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

pv.set_jupyter_backend(None)

warnings.filterwarnings(
    "ignore",
    category=ImportWarning
)

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
video_path = "nature.mp4"

watermark_path = "w11.png"

resize_dim = (256, 256)

alpha = 0.02

output_watermarked_path = "resnet_selected_watermarked.avi"

saved_frames_folder = "watermarked_frames"

selected_frames_file = "selected_frames.npy"

video_cube_file = "watermarked_video_cube.npy"

os.makedirs(
    saved_frames_folder,
    exist_ok=True
)

# -------------------------------------------------------------
# LOAD RESNET50
# -------------------------------------------------------------
print("\nLoading ResNet50...")

base_model = ResNet50(
    weights='imagenet',
    include_top=False
)

feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer(
        'conv4_block6_out'
    ).output
)

print("ResNet50 Loaded Successfully.")

# -------------------------------------------------------------
# RESNET50 FRAME SCORING
# -------------------------------------------------------------
print("\nComputing Frame Scores...")

temp_cap = cv2.VideoCapture(
    video_path
)

frame_scores = []

while True:

    ret, temp_frame = temp_cap.read()

    if not ret:
        break

    temp_frame = cv2.resize(
        temp_frame,
        resize_dim
    )

    rgb = cv2.cvtColor(
        temp_frame,
        cv2.COLOR_BGR2RGB
    )

    inp = cv2.resize(
        rgb,
        (224, 224)
    )

    inp = np.expand_dims(
        inp,
        axis=0
    )

    inp = preprocess_input(inp)

    features = feature_extractor.predict(
        inp,
        verbose=0
    )

    # Feature importance score
    score = np.mean(features)

    frame_scores.append(score)

temp_cap.release()

frame_scores = np.array(
    frame_scores
)

# -------------------------------------------------------------
# SELECT TOP 30% FRAMES
# -------------------------------------------------------------
num_selected = int(
    len(frame_scores) * 0.3
)

selected_frames = np.argsort(
    frame_scores
)[-num_selected:]

selected_frames = sorted(
    selected_frames.tolist()
)

selected_frames_set = set(
    selected_frames
)

np.save(
    selected_frames_file,
    np.array(selected_frames)
)

print(
    f"\nSelected "
    f"{len(selected_frames)} "
    f"frames for watermark embedding."
)

# -------------------------------------------------------------
# OPEN VIDEO AGAIN
# -------------------------------------------------------------
cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(
    cv2.CAP_PROP_FPS
)

fourcc = cv2.VideoWriter_fourcc(
    *'XVID'
)

out_watermarked = cv2.VideoWriter(
    output_watermarked_path,
    fourcc,
    fps,
    resize_dim
)

# -------------------------------------------------------------
# LOAD WATERMARK
# -------------------------------------------------------------
wm_color = cv2.imread(
    watermark_path
)

if wm_color is None:
    print("Error: Could not load watermark.")
    exit()

wm_color = cv2.resize(
    wm_color,
    (128, 128)
)

# -------------------------------------------------------------
# STORAGE
# -------------------------------------------------------------
watermarked_frames = []

psnr_list = []
ssim_list = []
ncc_list = []

frame_idx = 0

# -------------------------------------------------------------
# MAIN WATERMARKING LOOP
# -------------------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        resize_dim
    )

    original_frame = frame.copy()

    # ---------------------------------------------------------
    # WATERMARK ONLY SELECTED FRAMES
    # ---------------------------------------------------------
    if frame_idx in selected_frames_set:

        b_f, g_f, r_f = cv2.split(
            frame
        )

        b_wm, g_wm, r_wm = cv2.split(
            wm_color
        )

        watermarked_channels = []

        # -----------------------------------------------------
        # RGB DWT WATERMARKING
        # -----------------------------------------------------
        for f_c, w_c in zip(
            [b_f, g_f, r_f],
            [b_wm, g_wm, r_wm]
        ):

            f_c = np.float32(f_c)

            coeffs = pywt.dwt2(
                f_c,
                'haar'
            )

            LL, (LH, HL, HH) = coeffs

            w_resized = cv2.resize(
                w_c,
                (
                    LL.shape[1],
                    LL.shape[0]
                )
            )

            # -------------------------------------------------
            # CLASSICAL STABLE EMBEDDING
            # -------------------------------------------------
            LL_wm = LL + (
                alpha *
                w_resized *
                np.max(LL)
                / 255.0
            )

            coeffs_wm = (
                LL_wm,
                (LH, HL, HH)
            )

            wm_c = pywt.idwt2(
                coeffs_wm,
                'haar'
            )

            wm_c = np.clip(
                wm_c,
                0,
                255
            ).astype(np.uint8)

            watermarked_channels.append(
                wm_c
            )

        watermarked_frame = cv2.merge(
            watermarked_channels
        )

    else:

        # -----------------------------------------------------
        # NON-SELECTED FRAMES
        # -----------------------------------------------------
        watermarked_frame = frame.copy()

    # ---------------------------------------------------------
    # SAVE FRAME
    # ---------------------------------------------------------
    watermarked_frames.append(
        watermarked_frame.copy()
    )

    out_watermarked.write(
        watermarked_frame
    )

    cv2.imwrite(
        os.path.join(
            saved_frames_folder,
            f"frame_{frame_idx:04d}.png"
        ),
        watermarked_frame
    )

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    psnr = peak_signal_noise_ratio(
        original_frame,
        watermarked_frame
    )

    ssim = structural_similarity(
        original_frame,
        watermarked_frame,
        channel_axis=2
    )

    orig_gray = cv2.cvtColor(
        original_frame,
        cv2.COLOR_BGR2GRAY
    )

    wm_gray = cv2.cvtColor(
        watermarked_frame,
        cv2.COLOR_BGR2GRAY
    )

    numerator = np.sum(
        (orig_gray - orig_gray.mean()) *
        (wm_gray - wm_gray.mean())
    )

    denominator = np.sqrt(
        np.sum(
            (orig_gray - orig_gray.mean())**2
        ) *
        np.sum(
            (wm_gray - wm_gray.mean())**2
        )
    )

    ncc = numerator / (
        denominator + 1e-8
    )

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    ncc_list.append(ncc)

    print(
        f"Frame {frame_idx}: "
        f"PSNR={psnr:.2f} dB, "
        f"SSIM={ssim:.4f}, "
        f"NCC={ncc:.4f}"
    )

    frame_idx += 1

cap.release()

out_watermarked.release()

cv2.destroyAllWindows()

# -------------------------------------------------------------
# SAVE VIDEO CUBE
# -------------------------------------------------------------
video_cube = np.stack(
    watermarked_frames
)

np.save(
    video_cube_file,
    video_cube
)

print(
    "\nVideo cube saved successfully."
)

print(
    "Shape:",
    video_cube.shape
)

# -------------------------------------------------------------
# PLOT METRICS
# -------------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(
    psnr_list,
    label="PSNR"
)

plt.plot(
    ssim_list,
    label="SSIM"
)

plt.plot(
    ncc_list,
    label="NCC"
)

plt.xlabel("Frame Index")

plt.ylabel("Metric Value")

plt.title(
    "ResNet50 Frame Selection + DWT Watermarking"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# -------------------------------------------------------------
# 3D VIDEO VISUALIZATION
# -------------------------------------------------------------
print(
    "\nOpening 3D Video Cube..."
)

plotter = pv.Plotter()

spacing = 1

for i, frame in enumerate(
    watermarked_frames
):

    rgb_image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    texture = pv.numpy_to_texture(
        rgb_image
    )

    plane = pv.Plane(
        center=(128, 128, i * spacing),
        i_size=256,
        j_size=256,
        direction=(0, 0, 1)
    )

    plotter.add_mesh(
        plane,
        texture=texture
    )

plotter.add_axes()

plotter.show(
    window_size=[900, 700]
)

plotter.close()

del plotter

print(
    "\nEmbedding Completed Successfully."
)
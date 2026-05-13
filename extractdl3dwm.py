# =============================================================
# FULL EXTRACTION CODE
# ResNet50 Frame Selection + Classical DWT Extraction
# =============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
video_cube_path = "watermarked_video_cube.npy"

original_video_path = "nature.mp4"

watermark_path = "w11.png"

selected_frames_file = "selected_frames.npy"

resize_dim = (256, 256)

alpha = 0.02

output_folder = "extracted_watermarks"

os.makedirs(
    output_folder,
    exist_ok=True
)

# -------------------------------------------------------------
# LOAD ORIGINAL WATERMARK
# -------------------------------------------------------------
original_wm = cv2.imread(
    watermark_path
)

if original_wm is None:
    print("Error: Could not load watermark.")
    exit()

original_wm = cv2.resize(
    original_wm,
    (128, 128)
)

# -------------------------------------------------------------
# LOAD SELECTED FRAMES
# -------------------------------------------------------------
selected_frames = np.load(
    selected_frames_file
)

selected_frames = set(
    selected_frames.tolist()
)

print(
    f"\nLoaded "
    f"{len(selected_frames)} "
    f"selected frames."
)

# -------------------------------------------------------------
# LOAD VIDEO CUBE
# -------------------------------------------------------------
video_cube = np.load(
    video_cube_path
)

print(
    "\nVideo cube loaded."
)

print(
    "Shape:",
    video_cube.shape
)

# -------------------------------------------------------------
# OPEN ORIGINAL VIDEO
# -------------------------------------------------------------
cap_orig = cv2.VideoCapture(
    original_video_path
)

if not cap_orig.isOpened():
    print(
        "Error: Could not open original video."
    )
    exit()

# -------------------------------------------------------------
# STORAGE
# -------------------------------------------------------------
frame_idx = 0

ncc_list = []

psnr_list = []
ssim_list = []

extracted_watermarks = []

# -------------------------------------------------------------
# MAIN EXTRACTION LOOP
# -------------------------------------------------------------
while True:

    ret_orig, frame_orig = cap_orig.read()

    if not ret_orig:
        break

    if frame_idx >= len(video_cube):
        break

    frame_orig = cv2.resize(
        frame_orig,
        resize_dim
    )

    frame_wm = video_cube[frame_idx]

    # ---------------------------------------------------------
    # ONLY EXTRACT FROM SELECTED FRAMES
    # ---------------------------------------------------------
    if frame_idx in selected_frames:

        b_o, g_o, r_o = cv2.split(
            frame_orig
        )

        b_w, g_w, r_w = cv2.split(
            frame_wm
        )

        extracted_channels = []

        # -----------------------------------------------------
        # RGB WATERMARK EXTRACTION
        # -----------------------------------------------------
        for orig_c, wm_c in zip(
            [b_o, g_o, r_o],
            [b_w, g_w, r_w]
        ):

            orig_c = np.float32(orig_c)

            wm_c = np.float32(wm_c)

            # -------------------------------------------------
            # DWT
            # -------------------------------------------------
            LL_o, _ = pywt.dwt2(
                orig_c,
                'haar'
            )

            LL_w, _ = pywt.dwt2(
                wm_c,
                'haar'
            )

            # -------------------------------------------------
            # WATERMARK EXTRACTION
            # -------------------------------------------------
            max_LL_o = np.max(
                LL_o
            )

            extracted = (
                (LL_w - LL_o)
                *
                255.0
                /
                (
                    alpha *
                    max_LL_o
                    + 1e-8
                )
            )

            extracted = np.clip(
                extracted,
                0,
                255
            ).astype(np.uint8)

            extracted = cv2.resize(
                extracted,
                (128, 128)
            )

            extracted_channels.append(
                extracted
            )

        # -----------------------------------------------------
        # MERGE RGB WATERMARK
        # -----------------------------------------------------
        extracted_wm = cv2.merge(
            extracted_channels
        )

        extracted_watermarks.append(
            extracted_wm
        )

        # -----------------------------------------------------
        # SAVE
        # -----------------------------------------------------
        cv2.imwrite(
            os.path.join(
                output_folder,
                f"extracted_wm_{frame_idx:04d}.png"
            ),
            extracted_wm
        )

        # -----------------------------------------------------
        # NCC
        # -----------------------------------------------------
        orig_gray = cv2.cvtColor(
            original_wm,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        ext_gray = cv2.cvtColor(
            extracted_wm,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        numerator = np.sum(
            (orig_gray - orig_gray.mean())
            *
            (ext_gray - ext_gray.mean())
        )

        denominator = np.sqrt(
            np.sum(
                (orig_gray - orig_gray.mean())**2
            )
            *
            np.sum(
                (ext_gray - ext_gray.mean())**2
            )
        )

        ncc = numerator / (
            denominator + 1e-8
        )

        ncc_list.append(ncc)

        # -----------------------------------------------------
        # PSNR / SSIM
        # -----------------------------------------------------
        psnr = peak_signal_noise_ratio(
            frame_orig,
            frame_wm
        )

        ssim = structural_similarity(
            frame_orig,
            frame_wm,
            channel_axis=2
        )

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print(
            f"Frame {frame_idx} | "
            f"NCC={ncc:.4f} | "
            f"PSNR={psnr:.2f} | "
            f"SSIM={ssim:.4f}"
        )

    frame_idx += 1

cap_orig.release()

# -------------------------------------------------------------
# DISPLAY FINAL WATERMARK
# -------------------------------------------------------------
if len(extracted_watermarks) > 0:

    cv2.imshow(
        "Extracted Watermark",
        extracted_watermarks[-1]
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

else:

    print(
        "No watermark extracted."
    )

# -------------------------------------------------------------
# PLOT NCC
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.plot(
    ncc_list,
    label="NCC",
    color='red'
)

plt.xlabel(
    "Selected Frame Index"
)

plt.ylabel(
    "NCC"
)

plt.title(
    "Watermark Extraction NCC"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# -------------------------------------------------------------
# PLOT PSNR / SSIM
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.plot(
    psnr_list,
    label="PSNR",
    color='blue'
)

plt.plot(
    ssim_list,
    label="SSIM",
    color='green'
)

plt.xlabel(
    "Selected Frame Index"
)

plt.ylabel(
    "Metric Value"
)

plt.title(
    "PSNR and SSIM"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# -------------------------------------------------------------
# FINAL METRICS
# -------------------------------------------------------------
print("\n--------------------------------")

print(
    f"Average NCC  : "
    f"{np.mean(ncc_list):.4f}"
)

print(
    f"Average PSNR : "
    f"{np.mean(psnr_list):.2f} dB"
)

print(
    f"Average SSIM : "
    f"{np.mean(ssim_list):.4f}"
)

print("--------------------------------")

print(
    "\nExtraction Completed Successfully."
)
import numpy as np
import pandas as pd
import os
from skimage.restoration import denoise_wavelet

# --- Configuration ---
SUBJECT_FILE = r'F:\Amazon ML challenge\SPG\.venv\subject_seismic.npy'
OUTPUT_CSV = 'wavelet_denoised_transposed_output.csv'
RESULTS_DIR = './denoising_results_wavelet_advanced'

# WAVELET PARAMETERS
WAVELET_METHOD = 'BayesShrink'
WAVELET_WAVELET = 'coif1'  # smoother than db4, preserves subtle features
WAVELET_MODE = 'soft'       # soft thresholding to reduce noise
WAVELET_LEVEL = None        # automatic maximum decomposition


# ----------------------------------------------------------------------
# 1. Utility Functions
# ----------------------------------------------------------------------

def normalize(arr, axis=None):
    """Normalize array along given axis for stable processing."""
    mu = arr.mean(axis=axis, keepdims=True)
    sd = arr.std(axis=axis, keepdims=True) + 1e-9
    return (arr - mu) / sd, mu, sd


def denoise_wavelet_seismic(data, wavelet=WAVELET_WAVELET, method=WAVELET_METHOD,
                            mode=WAVELET_MODE, level=WAVELET_LEVEL):
    """
    Advanced Wavelet denoising for seismic data:
    - Trace-wise normalization (per column)
    - Optional global normalization
    - Multi-level decomposition
    """
    print("  -> Normalizing data trace-wise...")
    data_n, mu, sd = normalize(data, axis=0)  # trace-wise normalization

    # Optional global normalization for better contrast
    data_n, global_mu, global_sd = normalize(data_n)

    print(f"  -> Applying advanced denoise_wavelet (wavelet={wavelet}, method={method})...")
    filtered_n = denoise_wavelet(
        data_n,
        method=method,
        mode=mode,
        wavelet=wavelet,
        rescale_sigma=True,
        wavelet_levels=level
    )

    print("  -> De-normalizing data...")
    filtered_n = filtered_n * global_sd + global_mu
    filtered_n = filtered_n * sd + mu
    return filtered_n


def plot_seismic_sections(*args):
    """Placeholder for plotting function."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    plot_path = os.path.join(RESULTS_DIR, 'denoising_comparison_advanced_wavelet.png')
    print(f"Plot function skipped/simulated. Plot path: {plot_path}")


# ----------------------------------------------------------------------
# 2. Main Execution
# ----------------------------------------------------------------------

def main_wavelet_advanced():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("--- Starting Advanced Wavelet Denoising Pipeline ---")

    # 1. Load the Subject Data
    try:
        loaded_data = np.load(SUBJECT_FILE)
        if isinstance(loaded_data, np.lib.npyio.NpzFile):
            if 'arr_0' in loaded_data:
                subject_data_noisy = loaded_data['arr_0'].astype(np.float32)
            else:
                raise KeyError(f"Key 'arr_0' not found in .npz file. Keys: {list(loaded_data.keys())}")
        else:
            subject_data_noisy = loaded_data.astype(np.float32)

        current_shape = subject_data_noisy.shape
        print(f"Loaded Array Shape: {current_shape}")

    except Exception as e:
        print(f"\nFATAL ERROR during data loading: {type(e).__name__}: {e}. Aborting. ❌")
        return

    if current_shape != (227, 301):
        print(f"\nWARNING: Expected shape (227, 301), but found {current_shape}. Proceeding with caution.")

    # 2. Apply Advanced Wavelet Denoising
    denoised_section = denoise_wavelet_seismic(
        subject_data_noisy,
        wavelet=WAVELET_WAVELET,
        method=WAVELET_METHOD,
        mode=WAVELET_MODE,
        level=WAVELET_LEVEL
    )

    # 3. Save Results in the TRANSPOSED CSV Format
    if denoised_section.ndim < 2 or denoised_section.shape[1] == 0:
        print("\nWARNING: Cannot save in transposed format (data is not 2D or has zero columns).")
        return

    num_cols = denoised_section.shape[1]
    column_headers = [str(i) for i in range(num_cols)]

    submission_df = pd.DataFrame({'row_id': [i for i in range(denoised_section.shape[0])]})
    for i in range(num_cols):
        submission_df[column_headers[i]] = denoised_section[:, i]

    submission_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Advanced denoised CSV saved: '{OUTPUT_CSV}'")
    print(f"CSV Shape: {submission_df.shape} (227 rows + row_id, 301 columns of data)")

    # Plotting (simulated)
    plot_seismic_sections(subject_data_noisy, denoised_section)


if __name__ == '__main__':
    main_wavelet_advanced()

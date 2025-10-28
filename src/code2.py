# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
# from tensorflow.keras.models import Model
# import pandas as pd
# import os
# import glob
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
#
# # --- Configuration (CRITICAL: Corrected Path) ---
# # Set the DATA_DIR using a raw string (r'...') to handle the Windows path correctly.
# DATA_DIR = r'F:\Amazon ML challenge\SPG\.venv\Finals_SPGxSLB\training_traces\training_traces'
#
# SUBJECT_FILE = './subject_seismic.npz'  # Assuming subject file is an .npz containing a single array named 'arr_0'
# OUTPUT_CSV = 'denoised_submission.csv'
# BASE_NOISE_STD_DEV = 0.05  # Base level for synthetic Gaussian noise (in normalized space)
# IMG_SIZE = (128, 128)  # Patch size
# EPOCHS = 40  # Training iterations (Increase if possible)
# BATCH_SIZE = 32
# TRAINING_SAMPLES_PER_SECTION = 100  # Increased sampling
#
#
# # ----------------------------------------------------------------------
# # 1. Utility Functions: Metrics and Loss
# # ----------------------------------------------------------------------
#
# def ssim_loss(y_true, y_pred):
#     """Structural Similarity Index Loss (1 - SSIM) using normalized data range [-1, 1]."""
#     return 1.0 - tf.image.ssim(y_true, y_pred, max_val=2.0)
#
#
# def combined_loss(y_true, y_pred):
#     """Weights MSE and SSIM to prioritize structural preservation (SSIM)."""
#     mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
#     ssim_l = ssim_loss(y_true, y_pred)
#     # Higher weight on SSIM (0.7) ensures geological structures are preserved
#     return 0.3 * mse + 0.7 * ssim_l
#
#
# def calculate_psnr_unnorm(img_clean, img_noisy):
#     """Calculates PSNR in dB on the original (un-normalized) amplitude range."""
#     img_clean = img_clean.astype(np.float64)
#     img_noisy = img_noisy.astype(np.float64)
#     mse = np.mean((img_clean - img_noisy) ** 2)
#     if mse == 0: return 100
#     max_pixel = np.max(np.abs(img_clean))
#     return 10 * np.log10(max_pixel ** 2 / mse)
#
#
# def calculate_ssim_unnorm(img_clean, img_noisy):
#     """Calculates SSIM on the original (un-normalized) amplitude range."""
#     data_range = np.max(img_clean) - np.min(img_clean)
#     return ssim(img_clean, img_noisy, data_range=data_range, channel_axis=None)
#
#
# # ----------------------------------------------------------------------
# # 2. Improved U-Net Model Architecture
# # ----------------------------------------------------------------------
#
# def create_unet(input_shape):
#     """Defines a deeper U-Net model with 4 encoding levels."""
#     inputs = Input(input_shape)
#
#     # Encoder (Level 1)
#     conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs);
#     conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     # Encoder (Level 2)
#     conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1);
#     conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     # Encoder (Level 3)
#     conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2);
#     conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#     # Bottleneck (Level 4)
#     conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3);
#     conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
#
#     # Decoder (Level 3)
#     up5 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4));
#     merge5 = Concatenate(axis=-1)([conv3, up5])
#     conv5 = Conv2D(128, 3, activation='relu', padding='same')(merge5);
#     conv5 = Conv2D(128, 3, activation='relu', padding='same')(conv5)
#
#     # Decoder (Level 2)
#     up6 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5));
#     merge6 = Concatenate(axis=-1)([conv2, up6])
#     conv6 = Conv2D(64, 3, activation='relu', padding='same')(merge6);
#     conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv6)
#
#     # Decoder (Level 1)
#     up7 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6));
#     merge7 = Concatenate(axis=-1)([conv1, up7])
#     conv7 = Conv2D(32, 3, activation='relu', padding='same')(merge7);
#     conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)
#
#     # Output uses 'tanh' for output in [-1, 1]
#     outputs = Conv2D(1, 1, activation='tanh')(conv7)
#
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam', loss=combined_loss, metrics=['mse'])
#     return model
#
#
# # ----------------------------------------------------------------------
# # 3. Data Loading and Synthetic Noise Injection
# # ----------------------------------------------------------------------
#
# def load_and_preprocess_data(data_dir, base_noise_std_dev, num_patches_per_section):
#     """Loads .npy files, normalizes, and generates noisy/clean patches with adaptive noise."""
#     clean_patches = []
#     noisy_patches = []
#
#     # Correctly look for .npy files
#     filepaths = glob.glob(os.path.join(data_dir, '*.npy'))
#     print(f"Found {len(filepaths)} seismic sections (.npy) for training.")
#
#     for path in filepaths:
#         try:
#             # Load from .npy
#             section = np.load(path).astype(np.float32)
#         except Exception as e:
#             print(f"Error loading {path}: {e}. Skipping.")
#             continue
#
#         # 1. Normalize the section to [-1, 1] range
#         max_amplitude = np.max(np.abs(section))
#         if max_amplitude == 0: continue
#         clean_section_norm = section / max_amplitude
#
#         xlines, times = clean_section_norm.shape
#
#         for _ in range(num_patches_per_section):
#             # 2. Randomly sample patches
#             if xlines < IMG_SIZE[0] or times < IMG_SIZE[1]: continue
#             start_x = np.random.randint(0, xlines - IMG_SIZE[0] + 1)
#             start_t = np.random.randint(0, times - IMG_SIZE[1] + 1)
#
#             clean_patch = clean_section_norm[start_x:start_x + IMG_SIZE[0],
#                           start_t:start_t + IMG_SIZE[1]]
#
#             # 3. Inject synthetic noise with slight variation
#             current_noise_std = base_noise_std_dev * np.random.uniform(0.8, 1.2)
#             noise = np.random.normal(0, current_noise_std, clean_patch.shape)
#             noisy_patch = clean_patch + noise
#
#             # Clip the noisy patch to maintain the amplitude range [-1, 1]
#             noisy_patch = np.clip(noisy_patch, -1.0, 1.0)
#
#             clean_patches.append(clean_patch)
#             noisy_patches.append(noisy_patch)
#
#     X_train = np.array(noisy_patches, dtype=np.float32)
#     Y_train = np.array(clean_patches, dtype=np.float32)
#
#     # Reshape for Keras (patches, xlines, times, channels=1)
#     X_train = np.expand_dims(X_train, axis=-1)
#     Y_train = np.expand_dims(Y_train, axis=-1)
#
#     print(f"Generated {X_train.shape[0]} total training patches.")
#     return X_train, Y_train
#
#
# # ----------------------------------------------------------------------
# # 4. Plotting Function
# # ----------------------------------------------------------------------
#
# def plot_seismic_sections(noisy_section, denoised_section, title="Seismic Denoising Results"):
#     """
#     Plots the noisy input, the denoised output, and the noise estimate.
#     """
#     noise_estimate = noisy_section - denoised_section
#     vmax = np.max(np.abs(noisy_section)) * 0.95
#     vmin = -vmax
#
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#     fig.suptitle(title, fontsize=16)
#
#     # 1. Noisy Input
#     axes[0].imshow(noisy_section, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
#     axes[0].set_title('Noisy Input (Subject Section)')
#     axes[0].set_xlabel('Crossline (xline)');
#     axes[0].set_ylabel('Time Sample')
#     plt.colorbar(axes[0].images[0], ax=axes[0], orientation='vertical', label='Amplitude')
#
#     # 2. Denoised Output
#     axes[1].imshow(denoised_section, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
#     axes[1].set_title('Denoised Output (Prediction)')
#     axes[1].set_xlabel('Crossline (xline)')
#
#     # 3. Noise Estimate (Difference)
#     n_vmax = np.max(np.abs(noise_estimate)) * 0.5
#     axes[2].imshow(noise_estimate, aspect='auto', cmap='gray', vmin=-n_vmax, vmax=n_vmax)
#     axes[2].set_title('Estimated Noise (Input - Denoised)')
#     axes[2].set_xlabel('Crossline (xline)')
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()
#
#
# # ----------------------------------------------------------------------
# # 5. Main Execution
# # ----------------------------------------------------------------------
#
# def main():
#     # --- A. Load and Prepare Training Data ---
#     X_train, Y_train = load_and_preprocess_data(DATA_DIR, BASE_NOISE_STD_DEV, TRAINING_SAMPLES_PER_SECTION)
#     if X_train.size == 0:
#         # Check for the subject file if training failed, just in case
#         if not os.path.exists(SUBJECT_FILE):
#             print(
#                 "\nERROR: Training data failed to load AND subject file is missing. Please ensure data paths are correct.")
#         else:
#             print(
#                 "\nFATAL ERROR: Training data failed to load. Please verify the directory path and that the .npy files exist.")
#         return
#
#     # --- B. Build and Train Model ---
#     input_shape = (*IMG_SIZE, 1)
#     model = create_unet(input_shape)
#
#     print("Starting Model Training...")
#     model.fit(
#         X_train,
#         Y_train,
#         batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         validation_split=0.1,
#         verbose=1
#     )
#     print("Training Complete.")
#
#     # --- C. Predict on Subject Seismic Section ---
#     subject_data_noisy = np.load(SUBJECT_FILE)['arr_0'].astype(np.float32)
#     original_shape = subject_data_noisy.shape
#
#     # Normalize subject data
#     subject_max_amp = np.max(np.abs(subject_data_noisy))
#     subject_norm = subject_data_noisy / subject_max_amp
#
#     xlines, times = original_shape
#     stride = IMG_SIZE[0] // 2  # Overlapping stride for smoother prediction
#
#     denoised_accumulator = np.zeros(original_shape, dtype=np.float32)
#     denoised_counter = np.zeros(original_shape, dtype=np.float32)
#
#     print("Starting Denoising Prediction (Sliding Window with Overlap)...")
#
#     # Overlapping prediction loop for seamless stitching
#     for i in range(0, xlines - IMG_SIZE[0] + 1, stride):
#         for j in range(0, times - IMG_SIZE[1] + 1, stride):
#             block = subject_norm[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]]
#
#             # Predict
#             input_block = np.expand_dims(np.expand_dims(block, axis=0), axis=-1)
#             predicted_block = model.predict(input_block, verbose=0)[0, :, :, 0]
#
#             # Accumulate prediction and count overlaps
#             denoised_accumulator[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]] += predicted_block
#             denoised_counter[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]] += 1.0
#
#     # Average the accumulated predictions where counter > 0
#     denoised_section_norm = np.divide(denoised_accumulator, denoised_counter,
#                                       out=denoised_accumulator, where=denoised_counter != 0)
#
#     # 4. Re-scale back to original amplitude
#     denoised_section = denoised_section_norm * subject_max_amp
#
#     # --- D. Evaluation, Plotting, and Submission ---
#
#     # Calculate metrics (Self-reference on Noisy Input)
#     final_psnr = calculate_psnr_unnorm(subject_data_noisy, denoised_section)
#     final_ssim = calculate_ssim_unnorm(subject_data_noisy, denoised_section)
#
#     print(f"\n--- Evaluation Metrics (Compared to NOISY Input) ---")
#     print(f"PSNR: {final_psnr:.2f} dB")
#     print(f"SSIM: {final_ssim:.4f}")
#
#     # Plot the results
#     plot_seismic_sections(subject_data_noisy, denoised_section)
#
#     # --- E. Format and Save Submission ---
#     denoised_flat = denoised_section.flatten()
#     row_ids = [f'{i}' for i in range(len(denoised_flat))]
#
#     submission_df = pd.DataFrame({
#         'row_id': row_ids,
#         'amplitude': denoised_flat
#     })
#
#     submission_df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\n✅ Denoising Complete. Result saved to: {OUTPUT_CSV}")
#
#
# if __name__ == '__main__':
#     # --- Dummy Data Setup (Ensure this block is executed to run the code if real data is missing) ---
#     # This block is essential for the script to run without immediately failing if a file is missing.
#     # It creates temporary, synthetic files in the current directory.
#     if not glob.glob(os.path.join(DATA_DIR, '*.npy')) or not os.path.exists(SUBJECT_FILE):
#         print("Setting up dummy data for demonstration (Since real files were not immediately found)...")
#         if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
#
#         # Create 5 dummy training files (.npy)
#         for i in range(5):
#             L = 256
#             x = np.linspace(-1, 1, L);
#             t = np.linspace(0, 200, L);
#             X, T = np.meshgrid(x, t)
#             structure = (np.sin(0.1 * T) * np.exp(-0.01 * T) + np.sin(20 * X + 0.05 * T ** 2))
#             noise = np.random.normal(0, 0.5 * np.max(np.abs(structure)), (L, L))
#             dummy_data = (structure + noise) * 10000 / np.max(np.abs(structure))
#             np.save(os.path.join(DATA_DIR, f'train_{i}.npy'), dummy_data.astype(np.float32))
#
#         # Create a dummy subject file (.npz)
#         L = 512
#         x = np.linspace(-1, 1, L);
#         t = np.linspace(0, 400, L);
#         X, T = np.meshgrid(x, t)
#         structure = (np.sin(0.1 * T) * np.exp(-0.01 * T) + np.sin(30 * X + 0.02 * T ** 2))
#         heavy_noise = np.random.normal(0, 0.7 * np.max(np.abs(structure)), (L, L))
#         subject_data = (structure + heavy_noise) * 10000 / np.max(np.abs(structure))
#         np.savez(SUBJECT_FILE, arr_0=subject_data.astype(np.float32))
#         print(f"Dummy data created (5 .npy files in {DATA_DIR} and 1 {SUBJECT_FILE}).")
#
#     main()


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# --- Configuration ---
# CRITICAL: Path to the 200 .npy training files (using raw string for Windows path)
DATA_DIR = r'F:\Amazon ML challenge\SPG\.venv\Finals_SPGxSLB\training_traces\training_traces'

SUBJECT_FILE = './subject_seismic.npz'  # Assumes subject file is in the script's root folder
OUTPUT_CSV = 'denoised_submission.csv'
BASE_NOISE_STD_DEV = 0.05
IMG_SIZE = (128, 128)
EPOCHS = 10  # Reduced for faster execution as requested
BATCH_SIZE = 32
TRAINING_SAMPLES_PER_SECTION = 100

# --- File paths for saving results ---
RESULTS_DIR = './denoising_results'
DENOISED_NP_FILE = os.path.join(RESULTS_DIR, 'denoised_section.npy')
NOISY_NP_FILE = os.path.join(RESULTS_DIR, 'noisy_subject_section.npy')


# ----------------------------------------------------------------------
# 1. Utility Functions: Metrics and Loss
# ----------------------------------------------------------------------

def ssim_loss(y_true, y_pred):
    """Structural Similarity Index Loss (1 - SSIM) using normalized data range [-1, 1]."""
    # max_val=2.0 for data normalized to [-1, 1] range.
    return 1.0 - tf.image.ssim(y_true, y_pred, max_val=2.0)


def combined_loss(y_true, y_pred):
    """Weights MSE and SSIM. High SSIM weight ensures structural/frequency preservation."""
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim_l = ssim_loss(y_true, y_pred)
    # High SSIM weight (0.7) prioritizes the wave shape and structural integrity.
    return 0.3 * mse + 0.7 * ssim_l


def calculate_psnr_unnorm(img_clean, img_noisy):
    """Calculates PSNR in dB on the original (un-normalized) amplitude range."""
    img_clean = img_clean.astype(np.float64)
    img_noisy = img_noisy.astype(np.float64)
    mse = np.mean((img_clean - img_noisy) ** 2)
    if mse == 0: return 100
    max_pixel = np.max(np.abs(img_clean))
    return 10 * np.log10(max_pixel ** 2 / mse)


def calculate_ssim_unnorm(img_clean, img_noisy):
    """Calculates SSIM on the original (un-normalized) amplitude range."""
    data_range = np.max(img_clean) - np.min(img_clean)
    return ssim(img_clean, img_noisy, data_range=data_range, channel_axis=None)


# ----------------------------------------------------------------------
# 2. Improved U-Net Model Architecture
# ----------------------------------------------------------------------

def create_unet(input_shape):
    """Defines a deeper U-Net model with 4 encoding levels."""
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs);
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1);
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2);
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3);
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)

    # Decoder (Uses Skip Connections)
    up5 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4));
    merge5 = Concatenate(axis=-1)([conv3, up5])
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(merge5);
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5));
    merge6 = Concatenate(axis=-1)([conv2, up6])
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(merge6);
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6));
    merge7 = Concatenate(axis=-1)([conv1, up7])
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(merge7);
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)

    # Output uses 'tanh' to maintain true amplitude ratios and bipolarity.
    outputs = Conv2D(1, 1, activation='tanh')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=combined_loss, metrics=['mse'])
    return model


# ----------------------------------------------------------------------
# 3. Data Loading and Synthetic Noise Injection
# ----------------------------------------------------------------------

def load_and_preprocess_data(data_dir, base_noise_std_dev, num_patches_per_section):
    """Loads .npy files, normalizes, and generates noisy/clean patches with adaptive noise."""
    clean_patches = []
    noisy_patches = []

    filepaths = glob.glob(os.path.join(data_dir, '*.npy'))
    print(f"Found {len(filepaths)} seismic sections (.npy) for training.")

    if not filepaths:
        return np.array([]), np.array([])

    for path in filepaths:
        try:
            # Load from .npy
            section = np.load(path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}. Skipping.")
            continue

        # 1. Normalize the section to [-1, 1] range
        max_amplitude = np.max(np.abs(section))
        if max_amplitude == 0: continue
        clean_section_norm = section / max_amplitude

        xlines, times = clean_section_norm.shape

        for _ in range(num_patches_per_section):
            # 2. Randomly sample patches
            if xlines < IMG_SIZE[0] or times < IMG_SIZE[1]: continue
            start_x = np.random.randint(0, xlines - IMG_SIZE[0] + 1)
            start_t = np.random.randint(0, times - IMG_SIZE[1] + 1)

            clean_patch = clean_section_norm[start_x:start_x + IMG_SIZE[0],
                          start_t:start_t + IMG_SIZE[1]]

            # 3. Inject synthetic noise with slight variation
            current_noise_std = base_noise_std_dev * np.random.uniform(0.8, 1.2)
            noise = np.random.normal(0, current_noise_std, clean_patch.shape)
            noisy_patch = clean_patch + noise

            # Clip the noisy patch to maintain the amplitude range [-1, 1]
            noisy_patch = np.clip(noisy_patch, -1.0, 1.0)

            clean_patches.append(clean_patch)
            noisy_patches.append(noisy_patch)

    X_train = np.array(noisy_patches, dtype=np.float32)
    Y_train = np.array(clean_patches, dtype=np.float32)

    X_train = np.expand_dims(X_train, axis=-1)
    Y_train = np.expand_dims(Y_train, axis=-1)

    print(f"Generated {X_train.shape[0]} total training patches.")
    return X_train, Y_train


# ----------------------------------------------------------------------
# 4. Plotting Function (Saves plots and results)
# ----------------------------------------------------------------------

def plot_seismic_sections(noisy_section, denoised_section, title="Seismic Denoising Results"):
    """Plots the sections and saves the plot image."""

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    noise_estimate = noisy_section - denoised_section
    vmax = np.max(np.abs(noisy_section)) * 0.95
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(title, fontsize=16)

    # 1. Noisy Input
    axes[0].imshow(noisy_section, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    axes[0].set_title('Noisy Input (Subject Section)');
    axes[0].set_xlabel('Crossline (xline)');
    axes[0].set_ylabel('Time Sample')
    plt.colorbar(axes[0].images[0], ax=axes[0], orientation='vertical', label='Amplitude')

    # 2. Denoised Output
    axes[1].imshow(denoised_section, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised Output (Prediction)');
    axes[1].set_xlabel('Crossline (xline)')

    # 3. Noise Estimate (Difference)
    n_vmax = np.max(np.abs(noise_estimate)) * 0.5
    axes[2].imshow(noise_estimate, aspect='auto', cmap='gray', vmin=-n_vmax, vmax=n_vmax)
    axes[2].set_title('Estimated Noise (Input - Denoised)');
    axes[2].set_xlabel('Crossline (xline)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(RESULTS_DIR, 'denoising_comparison.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to: {plot_path}")


# ----------------------------------------------------------------------
# 5. Main Execution
# ----------------------------------------------------------------------

def main():
    # Setup results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # --- A. Load and Prepare Training Data ---
    X_train, Y_train = load_and_preprocess_data(DATA_DIR, BASE_NOISE_STD_DEV, TRAINING_SAMPLES_PER_SECTION)

    if X_train.size == 0:
        print("\nFATAL ERROR: Training data failed to load or no patches could be generated. Aborting.")
        return

    if not os.path.exists(SUBJECT_FILE):
        print(f"\nFATAL ERROR: Subject file not found at {SUBJECT_FILE}. Aborting prediction.")
        return

    # --- B. Build and Train Model ---
    input_shape = (*IMG_SIZE, 1)
    model = create_unet(input_shape)

    print(f"\nStarting Model Training for {EPOCHS} epochs...")
    model.fit(
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1
    )
    print("Training Complete.")

    # --- C. Predict on Subject Seismic Section ---
    subject_data_noisy = np.load(SUBJECT_FILE)['arr_0'].astype(np.float32)
    original_shape = subject_data_noisy.shape

    # Normalize subject data
    subject_max_amp = np.max(np.abs(subject_data_noisy))
    subject_norm = subject_data_noisy / subject_max_amp

    xlines, times = original_shape
    stride = IMG_SIZE[0] // 2

    denoised_accumulator = np.zeros(original_shape, dtype=np.float32)
    denoised_counter = np.zeros(original_shape, dtype=np.float32)

    print("Starting Denoising Prediction (Sliding Window with Overlap)...")

    # Overlapping prediction loop
    for i in range(0, xlines - IMG_SIZE[0] + 1, stride):
        for j in range(0, times - IMG_SIZE[1] + 1, stride):
            block = subject_norm[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]]

            input_block = np.expand_dims(np.expand_dims(block, axis=0), axis=-1)
            predicted_block = model.predict(input_block, verbose=0)[0, :, :, 0]

            denoised_accumulator[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]] += predicted_block
            denoised_counter[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]] += 1.0

    # Average the accumulated predictions
    denoised_section_norm = np.divide(denoised_accumulator, denoised_counter,
                                      out=denoised_accumulator, where=denoised_counter != 0)

    # Re-scale back to original amplitude (Preserves True Amplitude)
    denoised_section = denoised_section_norm * subject_max_amp

    # --- D. Evaluation, Plotting, and Saving Results ---

    # Calculate metrics (Self-reference on Noisy Input)
    final_psnr = calculate_psnr_unnorm(subject_data_noisy, denoised_section)
    final_ssIM = calculate_ssim_unnorm(subject_data_noisy, denoised_section)

    print(f"\n--- Evaluation Metrics (Compared to NOISY Input) ---")
    print(f"PSNR: {final_psnr:.2f} dB")
    print(f"SSIM: {final_ssIM:.4f}")

    # Save raw NumPy arrays
    np.save(NOISY_NP_FILE, subject_data_noisy)
    np.save(DENOISED_NP_FILE, denoised_section)
    print(f"Raw noisy and denoised arrays saved to {RESULTS_DIR}.")

    # Plot the results
    plot_seismic_sections(subject_data_noisy, denoised_section)

    # --- E. Format and Save Submission (Required Format) ---
    denoised_flat = denoised_section.flatten()
    row_ids = [f'{i}' for i in range(len(denoised_flat))]

    submission_df = pd.DataFrame({
        'row_id': row_ids,
        'amplitude': denoised_flat
    })

    # Save to CSV and print required output format example
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Denoising Complete. Submission saved to: {OUTPUT_CSV}")

    # Print the specific output format example requested
    print("\n--- Example of Denoised Output (First 5 Rows) ---")
    print("row_id\tamplitude")
    # Print the first 5 rows with high precision
    for i in range(5):
        print(f"{i}\t{denoised_flat[i]:.17f}")


if __name__ == '__main__':
    main()
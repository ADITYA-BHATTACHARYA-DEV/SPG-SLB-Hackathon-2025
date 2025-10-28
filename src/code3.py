import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import KFold  # For Cross-Validation

# --- Configuration (CRITICAL: Sampling for Refined Test) ---
DATA_DIR = r'F:\Amazon ML challenge\SPG\.venv\Finals_SPGxSLB\training_traces\training_traces'

SUBJECT_FILE = './subject_seismic.npz'
OUTPUT_CSV = 'denoised_submission.csv'
RESULTS_DIR = './denoising_results'
DENOISED_NP_FILE = os.path.join(RESULTS_DIR, 'denoised_section.npy')
NOISY_NP_FILE = os.path.join(RESULTS_DIR, 'noisy_subject_section.npy')

# SAMPLING PARAMETERS
SAMPLED_TRAINING_SECTIONS = 180  # Use 180 sections for CV training
HOLD_OUT_TEST_SECTIONS = 20  # Use 20 sections for explicit test set (180 + 20 = 200)

BASE_NOISE_STD_DEV = 0.05
IMG_SIZE = (128, 128)
EPOCHS = 10
BATCH_SIZE = 32
TRAINING_SAMPLES_PER_SECTION = 100
K_FOLDS = 3


# ----------------------------------------------------------------------
# 1. Utility Functions: Metrics and Loss (No Change)
# ----------------------------------------------------------------------

def ssim_loss(y_true, y_pred):
    """Structural Similarity Index Loss (1 - SSIM) using normalized data range [-1, 1]."""
    return 1.0 - tf.image.ssim(y_true, y_pred, max_val=2.0)


def combined_loss(y_true, y_pred):
    """Weights MSE and SSIM."""
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim_l = ssim_loss(y_true, y_pred)
    return 0.3 * mse + 0.7 * ssim_l


def calculate_psnr_unnorm(img_clean, img_noisy):
    """Calculates PSNR in dB on the original (un-normalized) amplitude range."""
    img_clean = img_clean.astype(np.float64);
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
# 2. U-Net Model Architecture (No Change)
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

    # Decoder
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

    outputs = Conv2D(1, 1, activation='tanh')(conv7)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=combined_loss, metrics=['mse'])
    return model


# ----------------------------------------------------------------------
# 3. Data Loading and Splitting (Modified for Sampling)
# ----------------------------------------------------------------------

def load_and_split_data(data_dir, num_train, num_test):
    """Loads all sections, shuffles, and splits into Training and Hold-out Test sets."""
    filepaths = glob.glob(os.path.join(data_dir, '*.npy'))
    print(f"Total found {len(filepaths)} seismic sections (.npy).")

    if len(filepaths) < (num_train + num_test):
        raise ValueError(
            f"Not enough data files found for splitting. Found {len(filepaths)}, need {num_train + num_test}.")

    # Shuffle filepaths to ensure random splits
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(filepaths)

    # Split filepaths
    train_val_paths = filepaths[:num_train]
    test_paths = filepaths[num_train:num_train + num_test]

    # Load sections based on file paths
    def load_sections_from_paths(paths):
        sections = []
        for path in paths:
            try:
                sections.append(np.load(path).astype(np.float32))
            except Exception as e:
                print(f"Error loading {path}: {e}. Skipping.")
        return sections

    train_val_sections = load_sections_from_paths(train_val_paths)
    test_sections = load_sections_from_paths(test_paths)

    print(f"Loaded {len(train_val_sections)} sections for Cross-Validation (Training/Validation).")
    print(f"Loaded {len(test_sections)} sections for Hold-Out Test.")

    return train_val_sections, test_sections


def generate_patches(sections, base_noise_std_dev, num_patches_per_section):
    """Generates noisy/clean patches from a list of sections."""
    clean_patches = []
    noisy_patches = []

    for section in sections:
        max_amplitude = np.max(np.abs(section))
        if max_amplitude == 0: continue
        clean_section_norm = section / max_amplitude
        xlines, times = clean_section_norm.shape

        for _ in range(num_patches_per_section):
            if xlines < IMG_SIZE[0] or times < IMG_SIZE[1]: continue
            start_x = np.random.randint(0, xlines - IMG_SIZE[0] + 1)
            start_t = np.random.randint(0, times - IMG_SIZE[1] + 1)

            clean_patch = clean_section_norm[start_x:start_x + IMG_SIZE[0],
                          start_t:start_t + IMG_SIZE[1]]

            current_noise_std = base_noise_std_dev * np.random.uniform(0.8, 1.2)
            noise = np.random.normal(0, current_noise_std, clean_patch.shape)
            noisy_patch = np.clip(clean_patch + noise, -1.0, 1.0)

            clean_patches.append(clean_patch)
            noisy_patches.append(noisy_patch)

    X = np.array(noisy_patches, dtype=np.float32)
    Y = np.array(clean_patches, dtype=np.float32)

    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)

    return X, Y


# ----------------------------------------------------------------------
# 4. Plotting Function (Modified to accept optional title)
# ----------------------------------------------------------------------

def plot_seismic_sections(noisy_section, denoised_section, plot_filename, title="Seismic Denoising Results"):
    """Plots the sections and saves the plot image."""

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    noise_estimate = noisy_section - denoised_section
    vmax = np.max(np.abs(noisy_section)) * 0.95
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(title, fontsize=16)

    axes[0].imshow(noisy_section, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    axes[0].set_title('Noisy Input');
    axes[0].set_xlabel('Crossline (xline)');
    axes[0].set_ylabel('Time Sample')
    plt.colorbar(axes[0].images[0], ax=axes[0], orientation='vertical', label='Amplitude')

    axes[1].imshow(denoised_section, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised Output');
    axes[1].set_xlabel('Crossline (xline)')

    n_vmax = np.max(np.abs(noise_estimate)) * 0.5
    axes[2].imshow(noise_estimate, aspect='auto', cmap='gray', vmin=-n_vmax, vmax=n_vmax)
    axes[2].set_title('Estimated Noise');
    axes[2].set_xlabel('Crossline (xline)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(RESULTS_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)  # Close figure to free up memory
    print(f"Plot saved to: {plot_path}")


# ----------------------------------------------------------------------
# 5. Main Execution with K-Fold Cross-Validation
# ----------------------------------------------------------------------

def main():
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    # 1. Load and Split Data
    try:
        train_val_sections, test_sections = load_and_split_data(
            DATA_DIR, SAMPLED_TRAINING_SECTIONS, HOLD_OUT_TEST_SECTIONS
        )
    except ValueError as e:
        print(f"\nFATAL ERROR during data loading: {e}. Aborting.")
        return

    # Prepare patches for the Hold-Out Test Set (used later for final model evaluation)
    X_holdout, Y_holdout = generate_patches(test_sections, BASE_NOISE_STD_DEV, TRAINING_SAMPLES_PER_SECTION)
    print(f"Generated {X_holdout.shape[0]} patches for Hold-Out Test.")

    # Convert the list of sections to a NumPy array of sections for KFold indexing
    train_val_sections_array = np.array(train_val_sections, dtype=object)

    # 2. K-Fold Cross-Validation Loop
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    best_model = None
    best_val_ssim = -np.inf

    print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation ---")

    for fold, (train_index, val_index) in enumerate(kf.split(train_val_sections_array)):
        print(f"\nTraining Fold {fold + 1}/{K_FOLDS}")

        # Select sections for current train and validation sets
        current_train_sections = train_val_sections_array[train_index].tolist()
        current_val_sections = train_val_sections_array[val_index].tolist()

        # Generate patches
        X_train, Y_train = generate_patches(current_train_sections, BASE_NOISE_STD_DEV, TRAINING_SAMPLES_PER_SECTION)
        X_val, Y_val = generate_patches(current_val_sections, BASE_NOISE_STD_DEV, TRAINING_SAMPLES_PER_SECTION)

        if X_train.size == 0 or X_val.size == 0:
            print(f"Skipping fold {fold + 1}: Insufficient patches generated.")
            continue

        # Build and train model
        model = create_unet((*IMG_SIZE, 1))

        model.fit(
            X_train, Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, Y_val),
            verbose=1
        )

        # Evaluate using SSIM on validation set
        val_predictions = model.predict(X_val, verbose=0)
        val_ssim_loss = ssim_loss(tf.convert_to_tensor(Y_val), tf.convert_to_tensor(val_predictions)).numpy()
        val_ssim = 1.0 - val_ssim_loss

        fold_scores.append(val_ssim)
        print(f"Fold {fold + 1} Results: Validation SSIM = {val_ssim:.4f}")

        # Track the best model
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            best_model = model
            print("--> NEW BEST MODEL FOUND")

    print("\n--- Cross-Validation Summary ---")
    print(f"Average Validation SSIM across {K_FOLDS} folds: {np.mean(fold_scores):.4f}")

    if best_model is None:
        print("\nFATAL ERROR: No models were successfully trained. Aborting final prediction.")
        return

    # 3. Evaluate BEST_MODEL on the Hold-Out Test Set
    if X_holdout.size > 0:
        print("\n--- Evaluating Best Model on Hold-Out Test Set ---")
        holdout_predictions = best_model.predict(X_holdout, verbose=0)
        holdout_ssim_loss = ssim_loss(tf.convert_to_tensor(Y_holdout),
                                      tf.convert_to_tensor(holdout_predictions)).numpy()
        holdout_ssim = 1.0 - holdout_ssim_loss

        print(f"Hold-Out Test SSIM (Generalization Test): {holdout_ssim:.4f}")

    # 4. Final Prediction on Subject Section
    print("\n--- Generating Final Prediction for Subject Section ---")

    if not os.path.exists(SUBJECT_FILE):
        print(f"\nFATAL ERROR: Subject file not found at {SUBJECT_FILE}. Cannot generate final prediction.")
        return

    subject_data_noisy = np.load(SUBJECT_FILE)['arr_0'].astype(np.float32)
    original_shape = subject_data_noisy.shape

    subject_max_amp = np.max(np.abs(subject_data_noisy))
    subject_norm = subject_data_noisy / subject_max_amp

    xlines, times = original_shape
    stride = IMG_SIZE[0] // 2

    denoised_accumulator = np.zeros(original_shape, dtype=np.float32)
    denoised_counter = np.zeros(original_shape, dtype=np.float32)

    # Overlapping prediction loop
    for i in range(0, xlines - IMG_SIZE[0] + 1, stride):
        for j in range(0, times - IMG_SIZE[1] + 1, stride):
            block = subject_norm[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]]

            input_block = np.expand_dims(np.expand_dims(block, axis=0), axis=-1)
            predicted_block = best_model.predict(input_block, verbose=0)[0, :, :, 0]

            denoised_accumulator[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]] += predicted_block
            denoised_counter[i:i + IMG_SIZE[0], j:j + IMG_SIZE[1]] += 1.0

    denoised_section_norm = np.divide(denoised_accumulator, denoised_counter,
                                      out=denoised_accumulator, where=denoised_counter != 0)

    denoised_section = denoised_section_norm * subject_max_amp

    # 5. Save Results
    final_psnr = calculate_psnr_unnorm(subject_data_noisy, denoised_section)
    final_ssIM = calculate_ssim_unnorm(subject_data_noisy, denoised_section)

    print(f"\n--- Final Prediction Metrics (vs. Noisy Input) ---")
    print(f"PSNR: {final_psnr:.2f} dB")
    print(f"SSIM: {final_ssIM:.4f}")

    np.save(NOISY_NP_FILE, subject_data_noisy)
    np.save(DENOISED_NP_FILE, denoised_section)
    plot_seismic_sections(subject_data_noisy, denoised_section, 'denoising_comparison.png')

    denoised_flat = denoised_section.flatten()
    row_ids = [f'{i}' for i in range(len(denoised_flat))]

    submission_df = pd.DataFrame({'row_id': row_ids, 'amplitude': denoised_flat})
    submission_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Denoising Complete. Submission saved to: {OUTPUT_CSV}")
    print("\n--- Example of Denoised Output (First 5 Rows) ---")
    print("row_id\tamplitude")
    for i in range(5):
        print(f"{i}\t{denoised_flat[i]:.17f}")


if __name__ == '__main__':
    # Set TensorFlow to only use necessary resources
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU setup failed: {e}")

    main()
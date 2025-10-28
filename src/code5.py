import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Reshape, BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os, glob
from skimage.restoration import denoise_wavelet
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = r'F:\Amazon ML challenge\SPG\.venv\Finals_SPGxSLB\training_traces\training_traces'
SUBJECT_FILE = './subject_seismic.npz'
OUTPUT_CSV_LONG = 'denoised_submission_long.csv'
OUTPUT_CSV_WIDE = 'denoised_submission_wide.csv'
RESULTS_DIR = './denoising_results_wavelet_regressor'

EPOCHS = 30
BATCH_SIZE = 32
TRAINING_SAMPLES_PER_SECTION = 200
IMG_SIZE = (64, 64)
WAVELET_METHOD = 'BayesShrink'
WAVELET_WAVELET = 'db1'

TOTAL_SECTIONS = 200
HOLD_OUT_TEST_SECTIONS = 20
SAMPLED_TRAINING_SECTIONS = TOTAL_SECTIONS - HOLD_OUT_TEST_SECTIONS
K_FOLDS = 3
QUICK_TEST_PATCH_COUNT = 100

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------

def normalize_section(section):
    max_amp = np.max(np.abs(section)) + 1e-9
    normalized = section / max_amp
    return normalized, max_amp

def denoise_wavelet_seismic(data):
    data_n, max_amp = normalize_section(data)
    filtered_n = denoise_wavelet(
        data_n, method=WAVELET_METHOD, mode='soft',
        wavelet=WAVELET_WAVELET, rescale_sigma=True)
    return filtered_n * max_amp, max_amp

def calculate_psnr_unnorm(img_clean, img_noisy):
    mse = np.mean((img_clean - img_noisy) ** 2)
    if mse == 0: return 100
    max_pixel = np.max(np.abs(img_clean))
    return 10 * np.log10(max_pixel ** 2 / mse)

def calculate_ssim_unnorm(img_clean, img_noisy):
    data_range = np.max(img_clean) - np.min(img_clean)
    return ssim(img_clean, img_noisy, data_range=data_range, channel_axis=None)

# ----------------------------------------------------------------------
# Improved CNN with Residual Learning
# ----------------------------------------------------------------------
def create_improved_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    res = Conv2D(1, 1, padding='same')(x)
    x = Add()([inputs, res])
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(input_shape[0]*input_shape[1], activation='linear')(x)
    outputs = Reshape(input_shape)(outputs)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ----------------------------------------------------------------------
# Data Loading and Patch Generation
# ----------------------------------------------------------------------
def load_and_split_data(data_dir, num_train, num_test):
    filepaths = glob.glob(os.path.join(data_dir, '*.npy'))
    np.random.seed(42); np.random.shuffle(filepaths)
    train_val_paths = filepaths[:num_train]
    test_paths = filepaths[num_train:num_train+num_test]
    def load_sections(paths):
        sections = []
        for path in paths:
            try: sections.append(np.load(path).astype(np.float32))
            except: pass
        return sections
    return load_sections(train_val_paths), load_sections(test_paths)

def generate_patches(sections, num_patches_per_section):
    X_wavelet_patches, Y_clean_patches = [], []
    for section in sections:
        section = np.array(section, dtype=np.float32)
        clean_norm, section_max = normalize_section(section)
        wavelet_denoised, _ = denoise_wavelet_seismic(section)
        wavelet_norm = wavelet_denoised / section_max
        xlines, times = clean_norm.shape
        for _ in range(num_patches_per_section):
            if xlines < IMG_SIZE[0] or times < IMG_SIZE[1]: continue
            start_x = np.random.randint(0, xlines - IMG_SIZE[0] + 1)
            start_t = np.random.randint(0, times - IMG_SIZE[1] + 1)
            X_wavelet_patches.append(wavelet_norm[start_x:start_x+IMG_SIZE[0], start_t:start_t+IMG_SIZE[1]])
            Y_clean_patches.append(clean_norm[start_x:start_x+IMG_SIZE[0], start_t:start_t+IMG_SIZE[1]])
    X = np.expand_dims(np.array(X_wavelet_patches, dtype=np.float32), axis=-1)
    Y = np.expand_dims(np.array(Y_clean_patches, dtype=np.float32), axis=-1)
    return X, Y

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_seismic_sections(noisy, denoised, plot_filename, title="Seismic Denoising Results"):
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    noise_est = noisy - denoised
    vmax = np.max(np.abs(noisy))*0.95
    vmin = -vmax
    fig, axes = plt.subplots(1,3,figsize=(18,6), sharey=True)
    fig.suptitle(title, fontsize=16)
    axes[0].imshow(noisy, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    axes[0].set_title('Noisy Input'); axes[0].set_xlabel('Crossline'); axes[0].set_ylabel('Time')
    plt.colorbar(axes[0].images[0], ax=axes[0], orientation='vertical')
    axes[1].imshow(denoised, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised Output'); axes[1].set_xlabel('Crossline')
    n_vmax = np.max(np.abs(noise_est))*0.5
    axes[2].imshow(noise_est, aspect='auto', cmap='gray', vmin=-n_vmax, vmax=n_vmax)
    axes[2].set_title('Estimated Noise'); axes[2].set_xlabel('Crossline')
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(os.path.join(RESULTS_DIR, plot_filename))
    plt.close(fig)

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    train_val_sections, test_sections = load_and_split_data(DATA_DIR, SAMPLED_TRAINING_SECTIONS, HOLD_OUT_TEST_SECTIONS)
    train_val_array = np.array(train_val_sections, dtype=object)

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    best_model, best_val_ssim = None, -np.inf
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_array)):
        print(f"\nTraining Fold {fold+1}/{K_FOLDS}")
        X_train, Y_train = generate_patches([train_val_array[i] for i in train_idx], TRAINING_SAMPLES_PER_SECTION)
        X_val, Y_val = generate_patches([train_val_array[i] for i in val_idx], TRAINING_SAMPLES_PER_SECTION)
        if X_train.size==0 or X_val.size==0: continue
        model = create_improved_cnn((*IMG_SIZE,1))
        model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=(X_val,Y_val), verbose=1, callbacks=[early_stop])
        val_pred = model.predict(X_val, verbose=0)
        val_ssim_scores = [calculate_ssim_unnorm(Y_val[i,...,0], val_pred[i,...,0]) for i in range(Y_val.shape[0])]
        val_ssim = np.mean(val_ssim_scores)
        print(f"Fold {fold+1} Validation SSIM: {val_ssim:.4f}")
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            best_model = model
            print("--> New Best Model Found")

    if not best_model or not os.path.exists(SUBJECT_FILE):
        print("No best model or subject file missing. Aborting.")
        return

    # Load subject data
    subject_data_noisy = np.load(SUBJECT_FILE)['arr_0'].astype(np.float32)
    subject_max_amp = np.max(np.abs(subject_data_noisy))
    wavelet_denoised, _ = denoise_wavelet_seismic(subject_data_noisy)
    wavelet_norm = wavelet_denoised / subject_max_amp

    xlines, times = subject_data_noisy.shape
    stride = IMG_SIZE[0] // 4
    final_denoised_norm = np.zeros_like(subject_data_noisy)
    counter = np.zeros_like(subject_data_noisy)

    for i in range(0, xlines - IMG_SIZE[0] + 1, stride):
        for j in range(0, times - IMG_SIZE[1] + 1, stride):
            block = wavelet_norm[i:i+IMG_SIZE[0], j:j+IMG_SIZE[1]]
            pred = best_model.predict(np.expand_dims(np.expand_dims(block, axis=0), axis=-1), verbose=0)[0,...,0]
            final_denoised_norm[i:i+IMG_SIZE[0], j:j+IMG_SIZE[1]] += pred
            counter[i:i+IMG_SIZE[0], j:j+IMG_SIZE[1]] += 1.0

    denoised_section_norm = np.divide(final_denoised_norm, counter, out=final_denoised_norm, where=counter!=0)
    denoised_section = denoised_section_norm * subject_max_amp

    plot_seismic_sections(subject_data_noisy, denoised_section, 'denoising_comparison_wavelet_cnn.png')

    # --- Save LONG format ---
    denoised_flat = denoised_section.flatten()
    row_ids_long = [str(i) for i in range(len(denoised_flat))]
    pd.DataFrame({'row_id': row_ids_long, 'amplitude': denoised_flat}).to_csv(OUTPUT_CSV_LONG, index=False)
    print(f"✅ Submission saved in long format: {OUTPUT_CSV_LONG}")

    # --- Save WIDE format ---
    columns = [str(i) for i in range(times)]
    row_ids_wide = [i for i in range(xlines)]
    df_wide = pd.DataFrame(denoised_section, index=row_ids_wide, columns=columns)
    df_wide.insert(0, 'row_id', row_ids_wide)
    df_wide.to_csv(OUTPUT_CSV_WIDE, index=False)
    print(f"✅ Submission saved in wide format: {OUTPUT_CSV_WIDE}")

if __name__=='__main__':
    tf.config.threading.set_inter_op_parallelism_threads(0)
    main()

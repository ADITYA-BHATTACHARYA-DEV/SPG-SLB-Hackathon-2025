"""
Optimized CNN + XGBoost hybrid pipeline for seismic denoising with ReLU-based semantic regression head.
Updated to handle .npy training files (≈200 total), compute SSIM/PSNR, and produce .npz + .csv submission outputs.
"""

import os
import glob
import random
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import pandas as pd

# ---------------------------
# Utilities
# ---------------------------

def load_npy_array(path: str) -> np.ndarray:
    arr = np.load(path)
    return arr.astype(np.float32)

def save_npz_array(path: str, arr: np.ndarray):
    np.savez_compressed(path, arr)

# ---------------------------
# Dataset
# ---------------------------

class SeismicDataset(Dataset):
    def __init__(self, file_paths: List[str], patch_size: Tuple[int, int] = (64, 64), stride: int = 32, augment: bool = True):
        self.file_paths = file_paths
        self.patch_h, self.patch_w = patch_size
        self.stride = stride
        self.augment = augment
        self.index = []

        for p in self.file_paths:
            arr = load_npy_array(p)
            h, w = arr.shape
            for top in range(0, h - self.patch_h + 1, self.stride):
                for left in range(0, w - self.patch_w + 1, self.stride):
                    self.index.append((p, top, left))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        p, top, left = self.index[idx]
        arr = load_npy_array(p)
        patch = arr[top:top + self.patch_h, left:left + self.patch_w]
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)

        if self.augment:
            if random.random() < 0.5:
                patch = np.flipud(patch)
            if random.random() < 0.5:
                patch = np.fliplr(patch)

        inp = torch.from_numpy(patch).unsqueeze(0).float()
        tgt = inp.clone()
        sem = torch.tensor([0.0], dtype=torch.float32)
        return inp, tgt, sem

# ---------------------------
# CNN Model (UNet + ReLU)
# ---------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class DenoiseUNet(nn.Module):
    def __init__(self, in_ch=1, base=32, semantic_dim=1):
        super().__init__()
        self.inc = ConvBlock(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.mid = ConvBlock(base*4, base*8)
        self.up2 = Up(base*8, base*4)
        self.up1 = Up(base*4, base*2)
        self.up0 = Up(base*2, base)
        self.outc = nn.Conv2d(base, 1, 1)
        self.sem_pool = nn.AdaptiveAvgPool2d((1,1))
        self.sem_fc = nn.Sequential(nn.Linear(base*8, base*4), nn.ReLU(), nn.Linear(base*4, semantic_dim))
        self.res_scale = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        xm = self.mid(x3)
        sem = self.sem_fc(self.sem_pool(xm).view(xm.size(0), -1))
        x = self.up2(xm, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)
        out = self.outc(x)
        return x + self.res_scale * out, sem, xm

# ---------------------------
# Training & Evaluation
# ---------------------------

def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.L1Loss()
    model.to(device)
    scaler = GradScaler()
    best_loss = float('inf')

    for e in range(epochs):
        model.train()
        total = 0
        for xb, yb, sb in tqdm(train_loader, desc=f"Epoch {e+1}"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                out, sem, _ = model(xb)
                loss = loss_fn(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * xb.size(0)
        scheduler.step()
        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {e+1}: train {total/len(train_loader.dataset):.5f} val {val_loss:.5f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate_model(model, val_loader, device):
    loss_fn = nn.L1Loss()
    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for xb, yb, sb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out, sem, _ = model(xb)
            loss_sum += loss_fn(out, yb).item() * xb.size(0)
    return loss_sum / len(val_loader.dataset)

# ---------------------------
# XGBoost Residual Corrector
# ---------------------------

def extract_features(model, files, device):
    model.eval()
    feats, resids = [], []
    for p in tqdm(files, desc='Extract feats'):
        arr = load_npy_array(p)
        arrn = (arr - arr.mean()) / (arr.std() + 1e-6)
        xb = torch.from_numpy(arrn).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out, sem, xm = model(xb)
        sem_np = xm.mean(dim=(2,3)).squeeze(0).cpu().numpy()
        res = (arr - out.squeeze().cpu().numpy()).mean()
        feats.append(sem_np)
        resids.append(res)
    return np.vstack(feats), np.array(resids).reshape(-1,1)

def train_xgb(X, y):
    xtr, xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    ytr_s, yva_s = scaler.fit_transform(ytr), scaler.transform(yva)
    dtr, dva = xgb.DMatrix(xtr, label=ytr_s), xgb.DMatrix(xva, label=yva_s)
    params = {'objective':'reg:squarederror','learning_rate':0.05,'max_depth':6,'subsample':0.8,'colsample_bytree':0.8,'tree_method':'hist'}
    bst = xgb.train(params, dtr, 1000, [(dtr,'train'),(dva,'val')], early_stopping_rounds=30, verbose_eval=False)
    bst.scaler = scaler
    return bst

# ---------------------------
# Inference
# ---------------------------

def predict_hybrid(model, xgb_model, subject_path, device):
    arr = load_npy_array(subject_path)
    arrn = (arr - arr.mean()) / (arr.std() + 1e-6)
    xb = torch.from_numpy(arrn).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out, sem, xm = model(xb)
    den = out.squeeze().cpu().numpy()
    pooled = xm.mean(dim=(2,3)).squeeze(0).cpu().numpy().reshape(1,-1)
    dmat = xgb.DMatrix(pooled)
    pred_scaled = xgb_model.predict(dmat)[0]
    offset = xgb_model.scaler.inverse_transform([[pred_scaled]])[0,0]
    den += offset
    return den

# ---------------------------
# Main
# ---------------------------

if __name__ == '__main__':
    train_folder = 'train/'
    subject_path = 'subject_seismic.npy'

    files = sorted(glob.glob(os.path.join(train_folder, '*.npy')))
    tr_files, va_files = train_test_split(files, test_size=0.15, random_state=42)

    train_ds = SeismicDataset(tr_files, patch_size=(64,64), stride=32, augment=True)
    val_ds = SeismicDataset(va_files, patch_size=(64,64), stride=32, augment=False)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenoiseUNet()
    model = train_model(model, train_dl, val_dl, device, epochs=50)

    X, y = extract_features(model, va_files, device)
    xgb_model = train_xgb(X, y)

    denoised = predict_hybrid(model, xgb_model, subject_path, device)
    save_npz_array('denoised_subject_seismic.npz', denoised)
    pd.DataFrame({'amplitude': denoised.flatten()}).to_csv('submission.csv', index_label='row_id')

    # SSIM / PSNR (optional: requires reference)
    # Here we compare subject noisy vs denoised for relative metric
    original = load_npy_array(subject_path)
    ssim_val = ssim(original, denoised, data_range=denoised.max()-denoised.min())
    psnr_val = psnr(original, denoised)
    print(f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")
    print('Saved denoised_subject_seismic.npz and submission.csv')

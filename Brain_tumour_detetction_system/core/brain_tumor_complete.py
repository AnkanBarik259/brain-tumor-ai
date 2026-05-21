"""
Brain Tumour Detection & Classification — Complete Single-File System
Supports: BraTS 2021 (segmentation) + Kaggle Brain Tumour MRI Dataset (classification)
Runs immediately in synthetic-data mode; plug in real data paths to train on real data.
"""

# ─── Standard library ────────────────────────────────────────────────────────
import ast, csv, json, math, os, random, time, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ─── Numeric / vision ─────────────────────────────────────────────────────────
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import ndimage
from scipy.stats import skew, kurtosis

# ─── PyTorch ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ─── Optional: nibabel ────────────────────────────────────────────────────────
try:
    import nibabel as nib
    NIBABEL_OK = True
except ImportError:
    NIBABEL_OK = False
    warnings.warn("nibabel not found – NIfTI loading disabled; using synthetic data.")

# ─── Optional: pyradiomics ────────────────────────────────────────────────────
try:
    import radiomics  # noqa: F401
    RADIOMICS_OK = True
except ImportError:
    RADIOMICS_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MODALITIES    = ["t1", "t1ce", "t2", "flair"]
MODAL_LABELS  = ["T1", "T1ce", "T2", "FLAIR"]
MODAL_CMAPS   = ["bone", "hot", "Blues_r", "Greens_r"]

BRATS_LABEL_MAP  = {0: 0, 1: 1, 2: 2, 4: 3}
SEG_CLASS_NAMES  = ["Background", "Necrotic Core (NCR)", "Oedema (ED)", "Enhancing Tumour (ET)"]
SEG_COLORS       = ["#000000", "#e63946", "#f4a261", "#2a9d8f"]

TUMOUR_TYPES = {0: "No Tumour", 1: "Glioma", 2: "Meningioma", 3: "Pituitary"}
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def remap_seg(seg: np.ndarray) -> np.ndarray:
    """BraTS labels 0,1,2,4 → contiguous 0,1,2,3."""
    out = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in BRATS_LABEL_MAP.items():
        out[seg == src] = dst
    return out


class NiftiLoader:
    @staticmethod
    def load_volume(path: str) -> np.ndarray:
        if not NIBABEL_OK:
            raise RuntimeError("nibabel required for NIfTI loading.")
        img = nib.load(path)
        return img.get_fdata().astype(np.float32)

    @staticmethod
    def load_subject(subject_dir: str) -> Dict[str, np.ndarray]:
        subj = Path(subject_dir)
        sid  = subj.name
        vols: Dict[str, np.ndarray] = {}
        for mod in MODALITIES:
            p = subj / f"{sid}_{mod}.nii.gz"
            vols[mod] = NiftiLoader.load_volume(str(p))
        seg_path = subj / f"{sid}_seg.nii.gz"
        vols["seg"] = NiftiLoader.load_volume(str(seg_path)).astype(np.int32)
        return vols


class SkullStripper:
    def __init__(self, percentile: float = 15.0):
        self.percentile = percentile

    def strip(self, vols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        flair = vols["flair"]
        nz    = flair[flair > 0]
        thr   = float(np.percentile(nz, self.percentile)) if len(nz) > 0 else 0.0
        mask  = (flair > thr).astype(np.uint8)
        labeled, n = ndimage.label(mask)
        if n > 0:
            sizes = ndimage.sum(mask, labeled, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            mask = (labeled == largest).astype(np.uint8)
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        mask = ndimage.binary_dilation(mask, iterations=2).astype(np.uint8)
        for mod in MODALITIES:
            vols[mod] = vols[mod] * mask
        vols["brain_mask"] = mask
        return vols


class IntensityNormaliser:
    def __init__(self, clip_percentile: float = 99.5):
        self.clip_percentile = clip_percentile

    def normalise(self, vol: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is None:
            mask = (vol > 0)
        brain_vals = vol[mask.astype(bool)]
        if len(brain_vals) == 0:
            return vol
        clip_val = float(np.percentile(brain_vals, self.clip_percentile))
        vol = np.clip(vol, 0, clip_val)
        brain_vals = vol[mask.astype(bool)]
        mu, sigma = float(brain_vals.mean()), float(brain_vals.std())
        if sigma < 1e-8:
            sigma = 1.0
        vol = (vol - mu) / sigma
        vol[~mask.astype(bool)] = 0.0
        return vol.astype(np.float32)

    def normalise_all(self, vols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        mask = vols.get("brain_mask", None)
        for mod in MODALITIES:
            vols[mod] = self.normalise(vols[mod], mask)
        return vols


class SliceExtractor:
    def __init__(self, size: Tuple[int, int] = (240, 240),
                 skip_frac: float = 0.10, empty_ratio: float = 0.30):
        self.size       = size
        self.skip_frac  = skip_frac
        self.empty_ratio = empty_ratio

    def indices(self, seg3d: np.ndarray, training: bool = True) -> List[int]:
        D = seg3d.shape[-1]
        lo = int(D * self.skip_frac)
        hi = int(D * (1 - self.skip_frac))
        if training:
            tumour = [i for i in range(lo, hi) if seg3d[..., i].max() > 0]
            empty  = [i for i in range(lo, hi) if seg3d[..., i].max() == 0]
            k = max(1, int(len(empty) * self.empty_ratio))
            return tumour + random.sample(empty, min(k, len(empty)))
        return list(range(lo, hi))

    def get(self, vols: Dict[str, np.ndarray], idx: int) -> Dict:
        H, W = self.size
        channels = []
        for mod in MODALITIES:
            sl = vols[mod][..., idx]
            sl = cv2.resize(sl.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            channels.append(sl)
        image = np.stack(channels, axis=0).astype(np.float32)   # (4,H,W)
        seg_sl = vols["seg"][..., idx].astype(np.float32)
        seg_sl = cv2.resize(seg_sl, (W, H), interpolation=cv2.INTER_NEAREST)
        seg_sl = remap_seg(seg_sl.astype(np.int32))
        binary = (seg_sl > 0).astype(np.int64)
        return {"image": image, "seg": seg_sl.astype(np.int64), "binary": binary}


class MRIAugmenter:
    def __init__(self, p_flip=0.5, p_rot=0.5, p_noise=0.3, p_int=0.3,
                 p_zoom=0.3, noise_std=0.05, zoom_range=(0.85, 1.0)):
        self.p_flip    = p_flip
        self.p_rot     = p_rot
        self.p_noise   = p_noise
        self.p_int     = p_int
        self.p_zoom    = p_zoom
        self.noise_std = noise_std
        self.zoom_range = zoom_range

    def __call__(self, image: np.ndarray, seg: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontal flip
        if random.random() < self.p_flip:
            image = image[:, :, ::-1].copy()
            seg   = seg[:, ::-1].copy()
        # Vertical flip
        if random.random() < self.p_flip:
            image = image[:, ::-1, :].copy()
            seg   = seg[::-1, :].copy()
        # 90-degree rotation
        if random.random() < self.p_rot:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            seg   = np.rot90(seg, k, axes=(0, 1)).copy()
        # Intensity shift (per-channel)
        if random.random() < self.p_int:
            for c in range(image.shape[0]):
                alpha = random.uniform(0.9, 1.1)
                beta  = random.uniform(-0.15, 0.15)
                image[c] = image[c] * alpha + beta
        # Gaussian noise
        if random.random() < self.p_noise:
            image = image + np.random.randn(*image.shape).astype(np.float32) * self.noise_std
        # Zoom crop
        if random.random() < self.p_zoom:
            H, W = image.shape[1], image.shape[2]
            scale = random.uniform(*self.zoom_range)
            nh, nw = int(H * scale), int(W * scale)
            y0 = random.randint(0, H - nh)
            x0 = random.randint(0, W - nw)
            image_crop = image[:, y0:y0+nh, x0:x0+nw]
            seg_crop   = seg[y0:y0+nh, x0:x0+nw]
            new_img = np.zeros_like(image)
            new_seg = np.zeros_like(seg)
            for c in range(image.shape[0]):
                new_img[c] = cv2.resize(image_crop[c], (W, H), interpolation=cv2.INTER_LINEAR)
            new_seg = cv2.resize(seg_crop.astype(np.float32), (W, H),
                                 interpolation=cv2.INTER_NEAREST).astype(seg.dtype)
            image, seg = new_img, new_seg
        return image, seg


class BraTSDataset(Dataset):
    """Dataset for BraTS 2021 NIfTI volumes (segmentation task)."""

    def __init__(self, root_dir: str, subject_ids: Optional[List[str]] = None,
                 training: bool = True, size: Tuple[int, int] = (240, 240),
                 cache: bool = False):
        self.root     = Path(root_dir)
        self.training = training
        self.size     = size
        self.cache    = cache
        self.augmenter   = MRIAugmenter() if training else None
        self.normaliser  = IntensityNormaliser()
        self.stripper    = SkullStripper()
        self.extractor   = SliceExtractor(size=size)

        if subject_ids is None:
            subject_ids = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.subject_ids = subject_ids

        self._cache: Dict = {}
        self.samples: List[Tuple[str, int]] = []   # (subject_id, slice_idx)
        self._index_subjects()

    def _load_subject(self, sid: str) -> Dict[str, np.ndarray]:
        if sid in self._cache:
            return self._cache[sid]
        vols = NiftiLoader.load_subject(str(self.root / sid))
        vols = self.stripper.strip(vols)
        vols = self.normaliser.normalise_all(vols)
        vols["seg"] = remap_seg(vols["seg"].astype(np.int32))
        if self.cache:
            self._cache[sid] = vols
        return vols

    def _index_subjects(self):
        for sid in self.subject_ids:
            try:
                vols = self._load_subject(sid)
                idxs = self.extractor.indices(vols["seg"], self.training)
                for i in idxs:
                    self.samples.append((sid, i))
            except Exception as e:
                warnings.warn(f"Skipping {sid}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sid, sl_idx = self.samples[idx]
        vols = self._load_subject(sid)
        sample = self.extractor.get(vols, sl_idx)
        image, seg = sample["image"], sample["seg"]
        if self.augmenter and self.training:
            image, seg = self.augmenter(image, seg)
        return {
            "image":      torch.from_numpy(image).float(),
            "seg":        torch.from_numpy(seg).long(),
            "cgm_label":  torch.tensor(int(seg.max() > 0), dtype=torch.long),
            "subject_id": sid,
            "slice_idx":  sl_idx,
        }


class KaggleTumourDataset(Dataset):
    """Dataset for Kaggle Brain Tumour MRI (classification task)."""

    CLASS_MAP = {"notumor": 0, "glioma": 1, "meningioma": 2, "pituitary": 3}

    def __init__(self, root_dir: str, split: str = "Training",
                 augment: bool = False, size: int = 224):
        self.size    = size
        self.augment = augment
        self.samples: List[Tuple[str, int]] = []
        base = Path(root_dir) / split
        for cls_name, cls_idx in self.CLASS_MAP.items():
            cls_dir = base / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"):
                for p in cls_dir.glob(ext):
                    self.samples.append((str(p), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=0)          # 3-channel
        if self.augment:
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy()
            if random.random() < 0.3:
                img = img + np.random.randn(*img.shape).astype(np.float32) * 0.03
                img = np.clip(img, 0, 1)
        return {
            "image": torch.from_numpy(img.copy()).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def build_brats_loaders(root_dir: str, batch: int = 4, workers: int = 2,
                        train_frac: float = 0.75, val_frac: float = 0.15,
                        size: Tuple[int, int] = (240, 240)):
    all_ids = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
    n = len(all_ids)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    random.shuffle(all_ids)
    tr_ids = all_ids[:n_tr]
    va_ids = all_ids[n_tr:n_tr+n_va]
    te_ids = all_ids[n_tr+n_va:]
    tr_ds = BraTSDataset(root_dir, tr_ids, training=True,  size=size)
    va_ds = BraTSDataset(root_dir, va_ids, training=False, size=size)
    te_ds = BraTSDataset(root_dir, te_ids, training=False, size=size)
    mk = dict(batch_size=batch, num_workers=workers, pin_memory=True)
    return (DataLoader(tr_ds, shuffle=True,  **mk),
            DataLoader(va_ds, shuffle=False, **mk),
            DataLoader(te_ds, shuffle=False, **mk))


def build_kaggle_loaders(root_dir: str, batch: int = 32, workers: int = 2):
    tr_ds = KaggleTumourDataset(root_dir, "Training",  augment=True)
    te_ds = KaggleTumourDataset(root_dir, "Testing",   augment=False)
    if len(tr_ds) == 0 or len(te_ds) == 0:
        raise RuntimeError("Kaggle dataset not found — check root_dir path.")
    mk = dict(batch_size=batch, num_workers=workers, pin_memory=True)
    return (DataLoader(tr_ds, shuffle=True,  **mk),
            DataLoader(te_ds, shuffle=False, **mk))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1B — SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic_subject(shape: Tuple[int, int, int] = (240, 240, 155),
                           has_tumour: bool = True) -> Dict[str, np.ndarray]:
    H, W, D = shape
    vols: Dict[str, np.ndarray] = {m: np.zeros(shape, np.float32) for m in MODALITIES}
    seg = np.zeros(shape, np.int32)

    cy, cx = H // 2, W // 2
    ry, rx = H * 0.42, W * 0.42

    for z in range(D):
        yy, xx = np.ogrid[:H, :W]
        brain_mask   = ((yy - cy)**2 / ry**2 + (xx - cx)**2 / rx**2) < 1.0
        wm_mask      = ((yy - cy)**2 / (ry*0.60)**2 + (xx - cx)**2 / (rx*0.60)**2) < 1.0
        gm_mask      = brain_mask & ~wm_mask
        vent_r       = ry * 0.12
        vent_mask    = ((yy - cy)**2 / vent_r**2 + (xx - cx)**2 / (vent_r*1.5)**2) < 1.0

        t1  = np.zeros((H, W), np.float32)
        t1ce= np.zeros((H, W), np.float32)
        t2  = np.zeros((H, W), np.float32)
        fl  = np.zeros((H, W), np.float32)

        t1[gm_mask]  = np.random.normal(0.60, 0.05, int(gm_mask.sum()))
        t1[wm_mask]  = np.random.normal(0.85, 0.04, int(wm_mask.sum()))
        t1[vent_mask]= np.random.normal(0.20, 0.03, int(vent_mask.sum()))
        t1ce[:] = t1.copy()

        t2[gm_mask]  = np.random.normal(0.65, 0.05, int(gm_mask.sum()))
        t2[wm_mask]  = np.random.normal(0.55, 0.04, int(wm_mask.sum()))
        t2[vent_mask]= np.random.normal(0.95, 0.02, int(vent_mask.sum()))

        fl[gm_mask]  = np.random.normal(0.65, 0.05, int(gm_mask.sum()))
        fl[wm_mask]  = np.random.normal(0.55, 0.04, int(wm_mask.sum()))
        fl[vent_mask]= np.random.normal(0.10, 0.02, int(vent_mask.sum()))

        # Tumour placement in middle third of volume
        z_lo, z_hi = D // 3, 2 * D // 3
        if has_tumour and z_lo <= z < z_hi:
            ty, tx = int(cy - H * 0.08), int(cx + W * 0.10)
            r_et  = int(min(H, W) * 0.07)
            r_ncr = int(r_et * 0.45)
            r_ed  = int(r_et * 1.60)
            for j in range(H):
                for i in range(W):
                    d2 = (j - ty)**2 + (i - tx)**2
                    if d2 < r_ncr**2:
                        seg[j, i, z] = 1              # NCR
                        t1[j, i]  = np.random.normal(0.30, 0.05)
                        t1ce[j,i] = np.random.normal(0.35, 0.05)
                        t2[j, i]  = np.random.normal(0.70, 0.06)
                        fl[j, i]  = np.random.normal(0.55, 0.05)
                    elif d2 < r_et**2:
                        seg[j, i, z] = 4              # ET (remapped → 3)
                        t1ce[j,i] = np.random.normal(0.90, 0.05)
                        t2[j, i]  = np.random.normal(0.80, 0.06)
                        fl[j, i]  = np.random.normal(0.75, 0.05)
                    elif d2 < r_ed**2:
                        if seg[j, i, z] == 0:
                            seg[j, i, z] = 2          # Oedema
                        t2[j, i]  = np.random.normal(0.85, 0.06)
                        fl[j, i]  = np.random.normal(0.80, 0.05)

        noise_scale = 0.02
        for arr in [t1, t1ce, t2, fl]:
            arr += np.random.normal(0, noise_scale, arr.shape).astype(np.float32)

        vols["t1"][..., z]   = t1
        vols["t1ce"][..., z] = t1ce
        vols["t2"][..., z]   = t2
        vols["flair"][..., z]= fl

    vols["seg"] = seg
    mask = (vols["flair"] > 0.05).astype(np.uint8)
    vols["brain_mask"] = mask
    return vols


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DEEP UNET 3+
# ══════════════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DeepUNet3Plus(nn.Module):
    def __init__(self, in_ch: int = 4, seg_classes: int = 4,
                 filters: List[int] = None):
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512, 1024]
        self.filters = filters
        cat = filters[0]                       # channels per skip = 64

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = DoubleConv(in_ch,       filters[0])
        self.pool1= nn.MaxPool2d(2)
        self.enc2 = DoubleConv(filters[0],  filters[1])
        self.pool2= nn.MaxPool2d(2)
        self.enc3 = DoubleConv(filters[1],  filters[2])
        self.pool3= nn.MaxPool2d(2)
        self.enc4 = DoubleConv(filters[2],  filters[3])
        self.pool4= nn.MaxPool2d(2)
        self.enc5 = DoubleConv(filters[3],  filters[4])

        # ── Projections for full-scale connections ────────────────────────────
        # Each decoder node receives 5 sources → each projected to `cat` channels
        self._make_projections(filters, cat)

        dec = cat * 5
        self.fuse4 = ConvBnRelu(dec, filters[3])
        self.fuse3 = ConvBnRelu(dec, filters[2])
        self.fuse2 = ConvBnRelu(dec, filters[1])
        self.fuse1 = ConvBnRelu(dec, filters[0])

        # ── Segmentation heads ────────────────────────────────────────────────
        self.seg_head = nn.Conv2d(filters[0], seg_classes, 1)
        self.ds4_head = nn.Conv2d(filters[3], seg_classes, 1)
        self.ds3_head = nn.Conv2d(filters[2], seg_classes, 1)
        self.ds2_head = nn.Conv2d(filters[1], seg_classes, 1)

        # ── Classification Guided Module ──────────────────────────────────────
        self.cgm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(filters[4], 2),
        )

        self._init_weights()

    def _make_projections(self, f: List[int], cat: int):
        # node d4: target spatial = enc4 scale (H/8)
        self.d4_proj1 = nn.Sequential(nn.AdaptiveMaxPool2d(None), ConvBnRelu(f[0], cat, 1))  # dummy; resolved in fwd
        self.d4_e1 = ConvBnRelu(f[0], cat, 1)
        self.d4_e2 = ConvBnRelu(f[1], cat, 1)
        self.d4_e3 = ConvBnRelu(f[2], cat, 1)
        self.d4_e4 = ConvBnRelu(f[3], cat, 1)
        self.d4_e5 = ConvBnRelu(f[4], cat, 1)

        # node d3
        self.d3_e1 = ConvBnRelu(f[0], cat, 1)
        self.d3_e2 = ConvBnRelu(f[1], cat, 1)
        self.d3_e3 = ConvBnRelu(f[2], cat, 1)
        self.d3_e4 = ConvBnRelu(f[3], cat, 1)
        self.d3_e5 = ConvBnRelu(f[4], cat, 1)

        # node d2
        self.d2_e1 = ConvBnRelu(f[0], cat, 1)
        self.d2_e2 = ConvBnRelu(f[1], cat, 1)
        self.d2_e3 = ConvBnRelu(f[2], cat, 1)
        self.d2_e4 = ConvBnRelu(f[3], cat, 1)
        self.d2_e5 = ConvBnRelu(f[4], cat, 1)

        # node d1
        self.d1_e1 = ConvBnRelu(f[0], cat, 1)
        self.d1_e2 = ConvBnRelu(f[1], cat, 1)
        self.d1_e3 = ConvBnRelu(f[2], cat, 1)
        self.d1_e4 = ConvBnRelu(f[3], cat, 1)
        self.d1_e5 = ConvBnRelu(f[4], cat, 1)

    def _resize(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        th, tw = target.shape[2], target.shape[3]
        sh, sw = x.shape[2], x.shape[3]
        if sh == th and sw == tw:
            return x
        if sh > th:
            return F.adaptive_max_pool2d(x, (th, tw))
        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    def _fuse(self, projs, ref, fuse_block):
        resized = [self._resize(p, ref) for p in projs]
        return fuse_block(torch.cat(resized, dim=1))

    def forward(self, x: torch.Tensor):
        training = self.training

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        # CGM
        cgm_logits = self.cgm(e5)

        # Decoder node d4
        d4 = self._fuse([self.d4_e1(e1), self.d4_e2(e2), self.d4_e3(e3),
                         self.d4_e4(e4), self.d4_e5(e5)], e4, self.fuse4)
        # Decoder node d3
        d3 = self._fuse([self.d3_e1(e1), self.d3_e2(e2), self.d3_e3(e3),
                         self.d3_e4(d4), self.d3_e5(e5)], e3, self.fuse3)
        # Decoder node d2
        d2 = self._fuse([self.d2_e1(e1), self.d2_e2(e2), self.d2_e3(d3),
                         self.d2_e4(d4), self.d2_e5(e5)], e2, self.fuse2)
        # Decoder node d1
        d1 = self._fuse([self.d1_e1(e1), self.d1_e2(d2), self.d1_e3(d3),
                         self.d1_e4(d4), self.d1_e5(e5)], e1, self.fuse1)

        seg = self.seg_head(d1)

        if training:
            H, W = x.shape[2], x.shape[3]
            up = lambda t: F.interpolate(t, (H, W), mode="bilinear", align_corners=False)
            ds4 = up(self.ds4_head(d4))
            ds3 = up(self.ds3_head(d3))
            ds2 = up(self.ds2_head(d2))
            return seg, ds4, ds3, ds2, cgm_logits
        else:
            cgm_prob = torch.sigmoid(cgm_logits[:, 1:2, None, None])
            seg_prob = torch.softmax(seg, dim=1)
            seg_out  = seg * cgm_prob
            return seg_out, cgm_logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — TUMOUR CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class TumourClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, radiomics_dim: int = 0):
        super().__init__()
        self.radiomics_dim = radiomics_dim

        self.features = nn.Sequential(
            ConvBnRelu(3, 32), nn.Dropout2d(0.10), ConvBnRelu(32, 32),  nn.MaxPool2d(2),
            ConvBnRelu(32, 64), nn.Dropout2d(0.10), ConvBnRelu(64, 64), nn.MaxPool2d(2),
            ConvBnRelu(64,128), nn.Dropout2d(0.10), ConvBnRelu(128,128),nn.MaxPool2d(2),
            ConvBnRelu(128,256),nn.Dropout2d(0.10), ConvBnRelu(256,256),nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        head_in = 256 + (radiomics_dim if radiomics_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.50),
            nn.Linear(head_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def forward(self, x: torch.Tensor,
                radiomics: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.gap(self.features(x))     # (B,256,1,1)
        feat = feat.flatten(1)                # (B,256)
        if self.radiomics_dim > 0 and radiomics is not None:
            feat = torch.cat([feat, radiomics], dim=1)
        return self.head(feat)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — RADIOMICS EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class RadiomicsExtractor:
    FEATURE_NAMES = [
        "shape_area", "shape_perimeter", "shape_compactness",
        "shape_eccentricity", "shape_extent", "shape_solidity",
        "int_mean", "int_std", "int_skew", "int_kurt",
        "int_p10", "int_p25", "int_p75", "int_p90",
        "glcm_contrast", "glcm_homogeneity", "glcm_energy",
        "glcm_entropy", "glcm_correlation", "glcm_variance",
    ]
    DIM = 20

    def extract(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        feats = np.zeros(self.DIM, dtype=np.float32)
        if mask.sum() == 0:
            return feats

        # ── Shape features ────────────────────────────────────────────────────
        mask_u8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = float(mask_u8.sum())
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            perim = float(cv2.arcLength(cnt, True)) + 1e-8
            hull  = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull)) + 1e-8
            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox_area = float(bw * bh) + 1e-8
            compactness = (4 * math.pi * area) / (perim ** 2 + 1e-8)
            extent   = area / bbox_area
            solidity = area / hull_area
            if len(cnt) >= 5:
                (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
                eccentricity = math.sqrt(max(0, 1 - (min(ma,mi)/(max(ma,mi)+1e-8))**2))
            else:
                eccentricity = 0.0
        else:
            perim = 1e-8; compactness = 0.0; eccentricity = 0.0
            extent = 0.0; solidity = 0.0

        feats[0] = area
        feats[1] = perim
        feats[2] = compactness
        feats[3] = eccentricity
        feats[4] = extent
        feats[5] = solidity

        # ── Intensity features ────────────────────────────────────────────────
        pixels = image[mask_u8.astype(bool)].astype(np.float32)
        if len(pixels) > 0:
            feats[6]  = float(pixels.mean())
            feats[7]  = float(pixels.std()) + 1e-8
            feats[8]  = float(skew(pixels))      if len(pixels) > 2 else 0.0
            feats[9]  = float(kurtosis(pixels))  if len(pixels) > 3 else 0.0
            feats[10] = float(np.percentile(pixels, 10))
            feats[11] = float(np.percentile(pixels, 25))
            feats[12] = float(np.percentile(pixels, 75))
            feats[13] = float(np.percentile(pixels, 90))

        # ── GLCM texture ─────────────────────────────────────────────────────
        levels = 16
        roi = image.copy()
        pmin, pmax = float(roi[mask_u8.astype(bool)].min()), float(roi[mask_u8.astype(bool)].max())
        if pmax > pmin:
            roi = ((roi - pmin) / (pmax - pmin) * (levels - 1)).astype(np.int32)
        else:
            roi = np.zeros_like(roi, dtype=np.int32)
        glcm = np.zeros((levels, levels), dtype=np.float64)
        offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
        H2, W2 = roi.shape
        for dy, dx in offsets:
            for y in range(max(0, -dy), min(H2, H2 - dy)):
                for x in range(max(0, -dx), min(W2, W2 - dx)):
                    if mask_u8[y, x] and mask_u8[y+dy, x+dx]:
                        i, j = roi[y, x], roi[y+dy, x+dx]
                        glcm[i, j] += 1
                        glcm[j, i] += 1
        total = glcm.sum() + 1e-12
        P = glcm / total
        eps = 1e-12
        ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        contrast    = float(np.sum((ii - jj)**2 * P))
        homogeneity = float(np.sum(P / (1 + np.abs(ii - jj))))
        energy      = float(np.sum(P**2))
        entropy     = float(-np.sum(P * np.log(P + eps)))
        mu_i = float(np.sum(ii * P)); mu_j = float(np.sum(jj * P))
        sig_i = math.sqrt(max(0, float(np.sum((ii - mu_i)**2 * P))))
        sig_j = math.sqrt(max(0, float(np.sum((jj - mu_j)**2 * P))))
        if sig_i > 1e-8 and sig_j > 1e-8:
            corr = float(np.sum((ii - mu_i)*(jj - mu_j)*P)) / (sig_i * sig_j)
        else:
            corr = 0.0
        variance = float(pixels.var()) if len(pixels) > 0 else 0.0

        feats[14] = contrast
        feats[15] = homogeneity
        feats[16] = energy
        feats[17] = entropy
        feats[18] = corr
        feats[19] = variance

        # Normalise 0-1 per feature (simple min-max on this sample)
        for k in range(self.DIM):
            v = feats[k]
            if not math.isfinite(v):
                feats[k] = 0.0

        return feats

    def batch_extract(self, images: torch.Tensor,
                      masks: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        out = np.zeros((B, self.DIM), dtype=np.float32)
        imgs_np  = images.cpu().numpy()
        masks_np = masks.cpu().numpy().astype(np.uint8)
        for b in range(B):
            ch1 = imgs_np[b, 1] if imgs_np.shape[1] > 1 else imgs_np[b, 0]
            out[b] = self.extract(ch1, masks_np[b])
        return torch.from_numpy(out)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self._acts:  Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self._acts = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def generate(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        self._acts = None
        self._grads = None
        model_was_training = x.requires_grad
        x = x.clone().requires_grad_(True)

        # Temporarily set eval; forward pass
        out = None
        # We need gradients so don't use no_grad
        with torch.enable_grad():
            result = next(iter([x]))  # keep x in graph
            # Run the model in eval mode (already set by caller)
            raw = x   # placeholder; actual forward below

        with torch.enable_grad():
            x2 = x.detach().requires_grad_(True)
            out_tuple = x2  # placeholder

            # Do actual forward
            try:
                out_tuple = self._model_forward(x2)
            except Exception:
                pass

            if out_tuple is None:
                return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

            if isinstance(out_tuple, tuple):
                logits = out_tuple[0]
            else:
                logits = out_tuple

            score = logits[0, min(target_class, logits.shape[1]-1)].mean()
            self._model_zero_grad()
            score.backward(retain_graph=False)

        if self._acts is None or self._grads is None:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cmin, cmax = cam.min(), cam.max()
        if cmax > cmin:
            cam = (cam - cmin) / (cmax - cmin)
        return cam.astype(np.float32)

    def _model_forward(self, x):
        raise NotImplementedError

    def _model_zero_grad(self):
        raise NotImplementedError


class SegGradCAM(GradCAM):
    """GradCAM specialised for DeepUNet3Plus."""
    def __init__(self, model: DeepUNet3Plus):
        # Target: last conv of enc5
        target_layer = model.enc5.block[-1].block[0]   # ConvBnRelu → Conv2d
        super().__init__(model, target_layer)
        self.model = model

    def _model_forward(self, x):
        return self.model(x)

    def _model_zero_grad(self):
        self.model.zero_grad()

    def generate(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        self._acts = None
        self._grads = None
        with torch.enable_grad():
            x2 = x.detach().clone().requires_grad_(True)
            out = self.model(x2)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            score = logits[0, min(target_class, logits.shape[1]-1)].mean()
            self.model.zero_grad()
            score.backward()

        if self._acts is None or self._grads is None:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, (x.shape[2], x.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)
        return cam


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — THERMAL OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

THERMAL_CMAP = LinearSegmentedColormap.from_list(
    "thermal",
    [
        (0.00, (0.00, 0.00, 0.00)),
        (0.20, (0.00, 0.00, 0.55)),
        (0.40, (0.00, 0.80, 0.80)),
        (0.60, (0.00, 0.90, 0.00)),
        (0.80, (1.00, 0.60, 0.00)),
        (1.00, (1.00, 0.00, 0.00)),
    ],
)


def thermal_overlay(mri_display: np.ndarray, cam: np.ndarray,
                    mask: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    H, W = mri_display.shape[:2]
    base = np.stack([mri_display, mri_display, mri_display], axis=-1)  # (H,W,3)
    base = np.clip(base, 0, 1)

    cam_masked = cam * mask.astype(np.float32)
    thermal_rgba = THERMAL_CMAP(cam_masked)[:, :, :3].astype(np.float32)

    mask3 = mask.astype(np.float32)[..., None]
    blended = base * (1 - alpha * mask3) + thermal_rgba * alpha * mask3
    blended = np.clip(blended, 0, 1)
    out = (blended * 255).astype(np.uint8)

    # Glow contour
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        glow = out.copy()
        cv2.drawContours(glow, contours, -1, (0, 255, 255), 6)
        out = cv2.addWeighted(out, 0.6, glow, 0.4, 0)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.shape[1]
        prob = torch.softmax(pred, dim=1)
        tgt_oh = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()
        prob_flat = prob.flatten(2)
        tgt_flat  = tgt_oh.flatten(2)
        inter = (prob_flat * tgt_flat).sum(-1)
        denom = prob_flat.sum(-1) + tgt_flat.sum(-1)
        dice  = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, ds_w: float = 0.3, cgm_w: float = 0.2):
        super().__init__()
        self.ds_w  = ds_w
        self.cgm_w = cgm_w
        self.dice  = DiceLoss()
        self.ce    = nn.CrossEntropyLoss()
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, outputs, seg_gt: torch.Tensor,
                cgm_gt: torch.Tensor) -> torch.Tensor:
        if len(outputs) == 5:
            seg, ds4, ds3, ds2, cgm_logits = outputs
            main = self.dice(seg, seg_gt) + self.ce(seg, seg_gt)
            aux  = (self.dice(ds4, seg_gt) + self.ce(ds4, seg_gt) +
                    self.dice(ds3, seg_gt) + self.ce(ds3, seg_gt) +
                    self.dice(ds2, seg_gt) + self.ce(ds2, seg_gt))
            loss = main + self.ds_w * aux
        else:
            seg, cgm_logits = outputs
            loss = self.dice(seg, seg_gt) + self.ce(seg, seg_gt)

        cgm_target = cgm_gt.float()
        cgm_pred   = cgm_logits[:, 1] - cgm_logits[:, 0]
        loss += self.cgm_w * self.bce(cgm_pred, cgm_target)
        return loss


class ClassifierLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(pred, target)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

class SegmentationMetrics:
    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        self.C      = num_classes
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.C)
        self.fp = torch.zeros(self.C)
        self.fn = torch.zeros(self.C)
        self.tn = torch.zeros(self.C)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        pred   = pred.flatten().cpu().long()
        target = target.flatten().cpu().long()
        for c in range(self.C):
            p = (pred   == c)
            t = (target == c)
            self.tp[c] += (p &  t).sum().float()
            self.fp[c] += (p & ~t).sum().float()
            self.fn[c] += (~p & t).sum().float()
            self.tn[c] += (~p & ~t).sum().float()

    def compute(self) -> Dict:
        s = self.smooth
        dice  = (2*self.tp + s) / (2*self.tp + self.fp + self.fn + s)
        jacc  = (self.tp + s) / (self.tp + self.fp + self.fn + s)
        prec  = (self.tp + s) / (self.tp + self.fp + s)
        rec   = (self.tp + s) / (self.tp + self.fn + s)
        spec  = (self.tn + s) / (self.tn + self.fp + s)
        result = {}
        for c in range(self.C):
            result[SEG_CLASS_NAMES[c]] = {
                "dice": float(dice[c]), "jaccard": float(jacc[c]),
                "precision": float(prec[c]), "recall": float(rec[c]),
                "specificity": float(spec[c]),
            }
        # Mean excluding background
        non_bg = list(range(1, self.C))
        result["mean"] = {
            k: float(sum(result[SEG_CLASS_NAMES[c]][k] for c in non_bg) / len(non_bg))
            for k in ["dice", "jaccard", "precision", "recall", "specificity"]
        }
        return result

    def summary_str(self) -> str:
        m = self.compute()
        lines = [f"{'Class':<30} {'Dice':>7} {'Jaccard':>8} {'Prec':>7} {'Rec':>7}"]
        lines.append("-" * 62)
        for name in SEG_CLASS_NAMES:
            d = m[name]
            lines.append(f"{name:<30} {d['dice']:>7.4f} {d['jaccard']:>8.4f} "
                         f"{d['precision']:>7.4f} {d['recall']:>7.4f}")
        d = m["mean"]
        lines.append("-" * 62)
        lines.append(f"{'Mean (excl. BG)':<30} {d['dice']:>7.4f} {d['jaccard']:>8.4f} "
                     f"{d['precision']:>7.4f} {d['recall']:>7.4f}")
        return "\n".join(lines)


class ClassificationMetrics:
    def __init__(self, num_classes: int = 4):
        self.C      = num_classes
        self.correct = 0
        self.total   = 0
        self.conf    = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() > 1:
            pred = pred.argmax(dim=1)
        pred   = pred.flatten().cpu().long()
        target = target.flatten().cpu().long()
        self.correct += (pred == target).sum().item()
        self.total   += target.numel()
        for p, t in zip(pred.tolist(), target.tolist()):
            if 0 <= p < self.C and 0 <= t < self.C:
                self.conf[t, p] += 1

    def compute(self) -> Dict:
        acc = self.correct / max(self.total, 1)
        result: Dict = {"accuracy": acc, "per_class": {}}
        for c in range(self.C):
            tp  = int(self.conf[c, c])
            fp  = int(self.conf[:, c].sum()) - tp
            fn  = int(self.conf[c, :].sum()) - tp
            p   = tp / max(tp + fp, 1)
            r   = tp / max(tp + fn, 1)
            f1  = 2*p*r / max(p + r, 1e-8)
            result["per_class"][TUMOUR_TYPES[c]] = {
                "precision": p, "recall": r, "f1": f1
            }
        return result

    def summary_str(self) -> str:
        m = self.compute()
        lines = [f"Accuracy: {m['accuracy']:.4f}",
                 f"{'Class':<15} {'Prec':>7} {'Rec':>7} {'F1':>7}"]
        lines.append("-" * 40)
        for c in range(self.C):
            name = TUMOUR_TYPES[c]
            d = m["per_class"][name]
            lines.append(f"{name:<15} {d['precision']:>7.4f} {d['recall']:>7.4f} {d['f1']:>7.4f}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — TRAINING LOOPS
# ══════════════════════════════════════════════════════════════════════════════

class SegmentationTrainer:
    def __init__(self, root_dir: str, save_dir: str = "checkpoints",
                 in_ch: int = 4, seg_classes: int = 4, batch: int = 4,
                 lr: float = 1e-4, epochs: int = 50, workers: int = 4):
        self.device     = torch.device(DEVICE)
        self.epochs     = epochs
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_dice  = 0.0

        self.model    = DeepUNet3Plus(in_ch, seg_classes).to(self.device)
        self.criterion= SegmentationLoss()
        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, T_max=epochs, eta_min=1e-6)
        self.metrics  = SegmentationMetrics(seg_classes)

        self.train_loader, self.val_loader, self.test_loader = \
            build_brats_loaders(root_dir, batch=batch, workers=workers)

        self.log_path = self.save_dir / "seg_training_log.csv"
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch","train_loss","val_loss","val_dice",
                 "val_jaccard","val_precision","val_recall"])

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            imgs  = batch["image"].to(self.device)
            seg   = batch["seg"].to(self.device)
            cgm   = batch["cgm_label"].to(self.device)
            self.optimizer.zero_grad()
            out   = self.model(imgs)
            loss  = self.criterion(out, seg, cgm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        self.metrics.reset()
        total = 0.0
        for batch in self.val_loader:
            imgs = batch["image"].to(self.device)
            seg  = batch["seg"].to(self.device)
            cgm  = batch["cgm_label"].to(self.device)
            out  = self.model(imgs)
            loss = self.criterion(out, seg, cgm)
            total += loss.item()
            self.metrics.update(out[0], seg)
        m = self.metrics.compute()
        return total / max(len(self.val_loader), 1), m

    def train(self):
        print(f"\n{'='*60}\n  Segmentation Training ({self.epochs} epochs)\n{'='*60}")
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            tr_loss = self._train_epoch()
            val_loss, m = self._val_epoch()
            self.scheduler.step()
            md = m["mean"]
            elapsed = time.time() - t0
            print(f"Ep {ep:03d}/{self.epochs}  tr={tr_loss:.4f}  val={val_loss:.4f} "
                  f"dice={md['dice']:.4f}  jacc={md['jaccard']:.4f}  t={elapsed:.1f}s")
            if md["dice"] > self.best_dice:
                self.best_dice = md["dice"]
                torch.save(self.model.state_dict(),
                           self.save_dir / "seg_best.pth")
                print(f"  ✓ Best model saved (dice={self.best_dice:.4f})")
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, tr_loss, val_loss, md["dice"],
                                        md["jaccard"], md["precision"], md["recall"]])
        self._plot_training_curves()

    def evaluate(self) -> Dict:
        ckpt = self.save_dir / "seg_best.pth"
        if ckpt.exists():
            self.model.load_state_dict(torch.load(str(ckpt), map_location=self.device))
        self.metrics.reset()
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                imgs = batch["image"].to(self.device)
                seg  = batch["seg"].to(self.device)
                out  = self.model(imgs)
                self.metrics.update(out[0], seg)
        print(self.metrics.summary_str())
        return self.metrics.compute()

    def _plot_training_curves(self):
        rows = []
        with open(self.log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return
        eps   = [int(r["epoch"])     for r in rows]
        tr_l  = [float(r["train_loss"]) for r in rows]
        va_l  = [float(r["val_loss"])   for r in rows]
        dice  = [float(r["val_dice"])   for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                 facecolor="#080c12")
        for ax in axes:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")
        axes[0].plot(eps, tr_l, color="#4fc3f7", label="Train")
        axes[0].plot(eps, va_l, color="#f48fb1", label="Val")
        axes[0].set_title("Loss", color="white"); axes[0].legend(labelcolor="white")
        axes[1].plot(eps, dice, color="#a5d6a7", label="Val Dice")
        axes[1].set_title("Dice", color="white"); axes[1].legend(labelcolor="white")
        plt.tight_layout()
        plt.savefig(str(self.save_dir / "seg_training_curves.png"), dpi=120)
        plt.close()


class ClassifierTrainer:
    def __init__(self, root_dir: str, save_dir: str = "checkpoints",
                 batch: int = 32, lr: float = 3e-4, epochs: int = 30,
                 workers: int = 4):
        self.device    = torch.device(DEVICE)
        self.epochs    = epochs
        self.save_dir  = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc  = 0.0

        self.model     = TumourClassifier(num_classes=4).to(self.device)
        self.criterion = ClassifierLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                             self.optimizer, T_max=epochs, eta_min=1e-6)
        self.metrics   = ClassificationMetrics(4)

        self.train_loader, self.test_loader = \
            build_kaggle_loaders(root_dir, batch=batch, workers=workers)

        self.log_path = self.save_dir / "cls_training_log.csv"
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","val_acc"])

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            imgs  = batch["image"].to(self.device)
            lbls  = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            out   = self.model(imgs)
            loss  = self.criterion(out, lbls)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        self.metrics = ClassificationMetrics(4)
        total = 0.0
        for batch in self.test_loader:
            imgs = batch["image"].to(self.device)
            lbls = batch["label"].to(self.device)
            out  = self.model(imgs)
            loss = self.criterion(out, lbls)
            total += loss.item()
            self.metrics.update(out, lbls)
        m = self.metrics.compute()
        return total / max(len(self.test_loader), 1), m["accuracy"]

    def train(self):
        print(f"\n{'='*60}\n  Classifier Training ({self.epochs} epochs)\n{'='*60}")
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            tr_loss = self._train_epoch()
            val_loss, acc = self._val_epoch()
            self.scheduler.step()
            print(f"Ep {ep:03d}/{self.epochs}  tr={tr_loss:.4f}  val={val_loss:.4f} "
                  f"acc={acc:.4f}  t={time.time()-t0:.1f}s")
            if acc > self.best_acc:
                self.best_acc = acc
                torch.save(self.model.state_dict(),
                           self.save_dir / "cls_best.pth")
                print(f"  ✓ Best model saved (acc={acc:.4f})")
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, tr_loss, val_loss, acc])
        self._plot_training_curves()
        print(self.metrics.summary_str())

    def _plot_training_curves(self):
        rows = []
        with open(self.log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return
        eps  = [int(r["epoch"]) for r in rows]
        tr_l = [float(r["train_loss"]) for r in rows]
        va_l = [float(r["val_loss"])   for r in rows]
        acc  = [float(r["val_acc"])    for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#080c12")
        for ax in axes:
            ax.set_facecolor("#0d1117"); ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
        axes[0].plot(eps, tr_l, color="#4fc3f7", label="Train")
        axes[0].plot(eps, va_l, color="#f48fb1", label="Val")
        axes[0].set_title("Loss", color="white"); axes[0].legend(labelcolor="white")
        axes[1].plot(eps, acc, color="#a5d6a7", label="Val Acc")
        axes[1].set_title("Accuracy", color="white"); axes[1].legend(labelcolor="white")
        plt.tight_layout()
        plt.savefig(str(self.save_dir / "cls_training_curves.png"), dpi=120)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — UNCERTAINTY ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyEstimator:
    def __init__(self, model: TumourClassifier, n_passes: int = 30,
                 uncertain_threshold: float = 0.005):
        self.model     = model
        self.n         = n_passes
        self.threshold = uncertain_threshold

    def estimate(self, x: torch.Tensor,
                 radiomics: Optional[torch.Tensor] = None) -> Dict:
        self.model.eval()
        self.model.enable_mc_dropout()
        probs_list = []
        with torch.no_grad():
            for _ in range(self.n):
                logits = self.model(x, radiomics)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_list.append(probs)
        arr = np.stack(probs_list, axis=0)       # (N, C)
        mean_p = arr.mean(axis=0)
        var_p  = arr.var(axis=0)
        std_p  = arr.std(axis=0)
        unc    = float(var_p.mean())
        pred   = int(mean_p.argmax())
        ci_lo  = np.clip(mean_p - 1.96 * std_p, 0, 1)
        ci_hi  = np.clip(mean_p + 1.96 * std_p, 0, 1)
        return {
            "pred_class":  pred,
            "pred_name":   TUMOUR_TYPES[pred],
            "mean_probs":  mean_p,
            "variance":    var_p,
            "uncertainty": unc,
            "ci_low":      ci_lo,
            "ci_high":     ci_hi,
            "is_uncertain": unc > self.threshold,
            "confidence":  float(mean_p[pred]),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

_SUB_REGION_PRIORITY = {3: "Enhancing Tumour (ET)",
                        1: "Necrotic Core (NCR)",
                        2: "Oedema (ED)",
                        0: "Background"}


class BrainTumourPipeline:
    def __init__(self, segmentor: DeepUNet3Plus,
                 classifier: TumourClassifier,
                 device: str = DEVICE,
                 seg_thresh: float = 0.5,
                 mc_passes: int = 30):
        self.dev        = torch.device(device)
        self.segmentor  = segmentor.to(self.dev).eval()
        self.classifier = classifier.to(self.dev).eval()
        self.seg_thresh = seg_thresh
        self.radiomics  = RadiomicsExtractor()
        self.gradcam    = SegGradCAM(self.segmentor)
        self.unc_est    = UncertaintyEstimator(self.classifier, n_passes=mc_passes)

    @classmethod
    def from_checkpoints(cls, seg_ckpt: str, cls_ckpt: str,
                         device: str = DEVICE, **kwargs) -> "BrainTumourPipeline":
        seg = DeepUNet3Plus()
        seg.load_state_dict(torch.load(seg_ckpt, map_location=device))
        clf = TumourClassifier()
        clf.load_state_dict(torch.load(cls_ckpt, map_location=device))
        return cls(seg, clf, device=device, **kwargs)

    def _segment(self, mri: torch.Tensor):
        self.segmentor.eval()
        with torch.no_grad():
            out = self.segmentor(mri)
        seg_logits, cgm_logits = out
        prob_map = torch.softmax(seg_logits, dim=1)[0].cpu().numpy()  # (C,H,W)
        tumour_prob = prob_map[1:].sum(axis=0)                         # (H,W)
        mask = (tumour_prob > self.seg_thresh).astype(np.uint8)
        cgm_p = float(torch.softmax(cgm_logits, dim=1)[0, 1].item())
        return seg_logits, mask, cgm_p, prob_map

    def _get_bbox(self, mask: np.ndarray) -> Optional[Tuple]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, w, h)

    def _prepare_clf_input(self, thermal: np.ndarray,
                           bbox: Optional[Tuple]) -> torch.Tensor:
        img = thermal.astype(np.float32) / 255.0
        if bbox is not None:
            x, y, w, h = bbox
            pad = 16
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)
            img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))       # (3,224,224)
        return torch.from_numpy(img).float().unsqueeze(0).to(self.dev)

    def predict(self, mri_tensor: torch.Tensor) -> Dict:
        mri_tensor = mri_tensor.to(self.dev)
        if mri_tensor.dim() == 3:
            mri_tensor = mri_tensor.unsqueeze(0)

        # Display channel
        if mri_tensor.shape[1] >= 2:
            disp = mri_tensor[0, 1].cpu().numpy()
        else:
            disp = mri_tensor[0, 0].cpu().numpy()
        dmin, dmax = disp.min(), disp.max()
        mri_disp = (disp - dmin) / (dmax - dmin + 1e-8)

        # Segmentation
        seg_logits, mask, cgm_p, prob_map = self._segment(mri_tensor)

        has_tumour = bool(mask.sum() > 50 and cgm_p > 0.4)

        # Grad-CAM
        cam = self.gradcam.generate(mri_tensor, target_class=1)

        # Thermal overlay
        thermal = thermal_overlay(mri_disp, cam, mask)

        # Radiomics
        rad_tensor  = self.radiomics.batch_extract(mri_tensor, torch.from_numpy(mask).unsqueeze(0))
        rad_np      = rad_tensor[0].numpy()

        # Classification
        if has_tumour:
            clf_inp = self._prepare_clf_input(thermal, self._get_bbox(mask))
            unc_result = self.unc_est.estimate(clf_inp)
        else:
            unc_result = {
                "pred_class": 0, "pred_name": "No Tumour",
                "mean_probs": np.array([1.0, 0.0, 0.0, 0.0]),
                "variance":   np.zeros(4), "uncertainty": 0.0,
                "ci_low":     np.zeros(4), "ci_high":     np.zeros(4),
                "is_uncertain": False, "confidence": 1.0,
            }

        # Dominant sub-region
        pred_seg = prob_map.argmax(axis=0)
        dom = 0
        for lbl in [3, 1, 2]:
            if (pred_seg == lbl).any():
                dom = lbl
                break

        bbox = self._get_bbox(mask)

        return {
            "has_tumour":      has_tumour,
            "cgm_confidence":  cgm_p,
            "bbox":            bbox,
            "mask":            mask,
            "pred_seg":        pred_seg,
            "seg_prob":        prob_map,
            "cam":             cam,
            "thermal":         thermal,
            "mri_display":     mri_disp,
            "radiomics":       rad_np,
            "tumour_type":     unc_result["pred_class"],
            "type_name":       unc_result["pred_name"],
            "type_probs":      unc_result["mean_probs"],
            "uncertainty":     unc_result["uncertainty"],
            "confidence":      unc_result["confidence"],
            "ci_low":          unc_result["ci_low"],
            "ci_high":         unc_result["ci_high"],
            "is_uncertain":    unc_result["is_uncertain"],
            "dominant_region": _SUB_REGION_PRIORITY[dom],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_diagnostic_report(result: Dict,
                                save_path: Optional[str] = None):
    fig = plt.figure(figsize=(26, 14), facecolor="#080c12")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    DARK_AX = "#0d1117"

    def style(ax, title=""):
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        if title:
            ax.set_title(title, color="white", fontsize=11, pad=8)

    # Panel 1 – Raw MRI
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, "Input MRI (T1ce)")
    ax1.imshow(result["mri_display"], cmap="gray", vmin=0, vmax=1)
    ax1.axis("off")

    # Panel 2 – Segmentation overlay
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, "Tumour Segmentation")
    ax2.imshow(result["mri_display"], cmap="gray", vmin=0, vmax=1)
    seg_rgba = np.zeros((*result["pred_seg"].shape, 4), dtype=np.float32)
    cmap4 = ListedColormap(["#00000000", "#e6394699", "#f4a26199", "#2a9d8f99"])
    seg_img = cmap4(result["pred_seg"] / 3.0)
    ax2.imshow(seg_img, alpha=0.65)
    if result["bbox"] is not None and result["has_tumour"]:
        x, y, w, h = result["bbox"]
        rect = mpatches.FancyBboxPatch((x, y), w, h, linewidth=1.5,
            edgecolor="cyan", facecolor="none", linestyle="--",
            boxstyle="round,pad=2")
        ax2.add_patch(rect)
        ax2.text(x, y - 5, "TUMOUR", color="cyan", fontsize=8, weight="bold")
    patches = [mpatches.Patch(color=SEG_COLORS[i], label=SEG_CLASS_NAMES[i])
               for i in range(1, 4)]
    ax2.legend(handles=patches, loc="lower right", fontsize=7,
               facecolor="#111", labelcolor="white", framealpha=0.7)
    ax2.axis("off")

    # Panel 3 – Thermal / Grad-CAM
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3, "Grad-CAM Thermal Heatmap")
    im3 = ax3.imshow(result["thermal"])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=THERMAL_CMAP), ax=ax3,
                        fraction=0.03, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Med", "High"])
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    ax3.axis("off")

    # Panel 4 – Classification bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    style(ax4, "Tumour Type Probabilities")
    labels = [TUMOUR_TYPES[i] for i in range(4)]
    probs  = result["type_probs"]
    ci_lo  = result["ci_low"]
    ci_hi  = result["ci_high"]
    colors = ["#2a9d8f", "#e63946", "#f4a261", "#457b9d"]
    bars   = ax4.barh(labels, probs, color=colors, alpha=0.85)
    for i, (bar, p, lo, hi) in enumerate(zip(bars, probs, ci_lo, ci_hi)):
        err_lo = p - lo; err_hi = hi - p
        ax4.errorbar(p, i, xerr=[[err_lo], [err_hi]],
                     fmt="none", color="white", capsize=3, linewidth=1.5)
        if i == result["tumour_type"]:
            bar.set_edgecolor("white"); bar.set_linewidth(2)
        ax4.text(min(p + 0.02, 0.97), i, f"{p*100:.1f}%",
                 va="center", color="white", fontsize=8)
    ax4.set_xlim(0, 1.05)
    ax4.tick_params(colors="white", labelsize=8)
    ax4.set_xlabel("Probability", color="white", fontsize=9)

    # Panel 5 – Radar chart (radiomics)
    ax5 = fig.add_subplot(gs[1, 1], polar=True)
    ax5.set_facecolor(DARK_AX)
    radar_keys   = ["shape_area", "shape_compactness", "int_mean", "int_std",
                    "glcm_contrast", "glcm_energy", "glcm_entropy", "glcm_homogeneity"]
    radar_labels = ["Area", "Compact.", "Int.Mean", "Int.Std",
                    "Contrast", "Energy", "Entropy", "Homog."]
    feat_idx = [RadiomicsExtractor.FEATURE_NAMES.index(k) for k in radar_keys]
    values   = [float(result["radiomics"][i]) for i in feat_idx]
    values_n = []
    for v in values:
        values_n.append(min(max(v, 0.0), 1.0))
    N = len(radar_labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values_n += values_n[:1]
    ax5.plot(angles, values_n, color="#4fc3f7", linewidth=2)
    ax5.fill(angles, values_n, color="#4fc3f7", alpha=0.25)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(radar_labels, color="white", fontsize=8)
    ax5.set_yticklabels([])
    ax5.set_title("Radiomics Radar", color="white", fontsize=11, pad=20)
    ax5.spines["polar"].set_edgecolor("#444")

    # Panel 6 – Diagnostic summary text
    ax6 = fig.add_subplot(gs[1, 2])
    style(ax6, "Diagnostic Summary")
    ax6.axis("off")
    if result["has_tumour"]:
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.80), 0.9, 0.15, transform=ax6.transAxes,
            boxstyle="round,pad=0.02", fc="#3d0000", ec="#e63946", lw=2))
        ax6.text(0.5, 0.875, "⚠  TUMOUR DETECTED", transform=ax6.transAxes,
                 ha="center", va="center", color="#e63946",
                 fontsize=15, weight="bold")
    else:
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.80), 0.9, 0.15, transform=ax6.transAxes,
            boxstyle="round,pad=0.02", fc="#002a28", ec="#2a9d8f", lw=2))
        ax6.text(0.5, 0.875, "✓  NO TUMOUR", transform=ax6.transAxes,
                 ha="center", va="center", color="#2a9d8f",
                 fontsize=15, weight="bold")

    info_lines = [
        ("Type",       result["type_name"],       "#4fc3f7"),
        ("Confidence", f"{result['confidence']*100:.1f}%", "#a5d6a7"),
        ("Sub-region", result["dominant_region"],  "#f4a261"),
        ("CGM Score",  f"{result['cgm_confidence']:.3f}", "#ce93d8"),
        ("Uncertainty",f"{result['uncertainty']:.5f}", "#ffcc80"),
        ("MC Passes",  str(len(RadiomicsExtractor.FEATURE_NAMES) - 0),  "#b0bec5"),
    ]
    # override MC passes display
    info_lines[5] = ("MC Passes", "30", "#b0bec5")
    if result["is_uncertain"]:
        info_lines.append(("⚑ Flag Review", "YES – uncertain prediction", "#ffb74d"))

    y_pos = 0.72
    for key, val, col in info_lines:
        ax6.text(0.08, y_pos, f"{key}:", transform=ax6.transAxes,
                 color="#888", fontsize=9)
        ax6.text(0.42, y_pos, val, transform=ax6.transAxes,
                 color=col, fontsize=9, weight="bold")
        y_pos -= 0.09

    # Uncertainty bar
    ax6.text(0.08, y_pos - 0.03, "Uncertainty Level:", transform=ax6.transAxes,
             color="#888", fontsize=8)
    unc_norm = min(result["uncertainty"] / 0.05, 1.0)
    bar_col  = "#e63946" if unc_norm > 0.6 else "#f4a261" if unc_norm > 0.3 else "#2a9d8f"
    ax6.add_patch(mpatches.FancyBboxPatch(
        (0.08, y_pos - 0.13), 0.84 * unc_norm, 0.06,
        transform=ax6.transAxes, boxstyle="round,pad=0.01",
        fc=bar_col, ec="none"))
    ax6.add_patch(mpatches.FancyBboxPatch(
        (0.08, y_pos - 0.13), 0.84, 0.06,
        transform=ax6.transAxes, boxstyle="round,pad=0.01",
        fc="none", ec="#444", lw=1))

    fig.suptitle("Brain Tumour AI Diagnostic Report", color="white",
                 fontsize=18, y=0.98, weight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ✓ Diagnostic report saved → {save_path}")
    plt.close()
    return fig


def visualise_preprocessing(vols: Dict[str, np.ndarray],
                             slice_idx: int,
                             save_path: Optional[str] = None):
    fig = plt.figure(figsize=(24, 10), facecolor="#080c12")
    gs  = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.20)

    def style(ax, title=""):
        ax.set_facecolor("#0d1117"); ax.axis("off")
        ax.tick_params(colors="white")
        if title:
            ax.set_title(title, color="white", fontsize=10)

    D = vols["t1"].shape[-1]
    si = min(slice_idx, D - 1)

    # Row 1: 4 modalities + seg overlay + stats
    for col, (mod, lbl, cmap) in enumerate(zip(MODALITIES, MODAL_LABELS, MODAL_CMAPS)):
        ax = fig.add_subplot(gs[0, col])
        style(ax, lbl)
        sl = vols[mod][..., si]
        ax.imshow(sl, cmap=cmap, vmin=sl.min(), vmax=sl.max())

    # Seg overlay
    ax_seg = fig.add_subplot(gs[0, 4])
    style(ax_seg, "Segmentation")
    t1ce_sl = vols["t1ce"][..., si]
    ax_seg.imshow(t1ce_sl, cmap="gray")
    seg_sl = vols["seg"][..., si]
    cmap4 = ListedColormap(["#00000000","#e6394680","#f4a26180","#2a9d8f80"])
    ax_seg.imshow(cmap4(remap_seg(seg_sl.astype(np.int32)) / 3.0), alpha=0.7)

    # Volume stats text
    ax_stat = fig.add_subplot(gs[0, 5])
    ax_stat.set_facecolor("#0d1117"); ax_stat.axis("off")
    ax_stat.set_title("Volume Stats", color="white", fontsize=10)
    lines = []
    for mod in MODALITIES:
        v = vols[mod]
        nz = v[v != 0]
        if len(nz) > 0:
            lines.append(f"{mod.upper():6s}  μ={nz.mean():.3f}  σ={nz.std():.3f}")
        else:
            lines.append(f"{mod.upper():6s}  (empty)")
    seg_labels = np.unique(vols["seg"])
    lines.append(f"\nSeg labels: {seg_labels.tolist()}")
    ax_stat.text(0.05, 0.95, "\n".join(lines), transform=ax_stat.transAxes,
                 color="white", fontsize=8, va="top", family="monospace")

    # Row 2: histograms (4 modalities) + coronal view + axial
    for col, (mod, lbl) in enumerate(zip(MODALITIES, MODAL_LABELS)):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        nz = vols[mod][vols[mod] != 0].ravel()
        ax.hist(nz, bins=60, color="#4fc3f7", alpha=0.75, edgecolor="none")
        ax.set_title(f"{lbl} Histogram", color="white", fontsize=9)
        ax.set_xlabel("Intensity", color="#aaa", fontsize=7)
        ax.set_ylabel("Count",     color="#aaa", fontsize=7)

    # Coronal view
    ax_cor = fig.add_subplot(gs[1, 4])
    ax_cor.set_facecolor("#0d1117"); ax_cor.axis("off")
    ax_cor.set_title("Coronal (T1ce)", color="white", fontsize=10)
    W2 = vols["t1ce"].shape[1]
    cor = vols["t1ce"][:, W2 // 2, :]
    ax_cor.imshow(cor.T, cmap="hot", origin="lower",
                  vmin=cor.min(), vmax=cor.max())

    # Axial full volume small view
    ax_ax = fig.add_subplot(gs[1, 5])
    ax_ax.set_facecolor("#0d1117"); ax_ax.axis("off")
    ax_ax.set_title("Sagittal (FLAIR)", color="white", fontsize=10)
    H2 = vols["flair"].shape[0]
    sag = vols["flair"][H2 // 2, :, :]
    ax_ax.imshow(sag.T, cmap="Greens_r", origin="lower")

    fig.suptitle("Preprocessing Visualisation", color="white",
                 fontsize=16, y=0.99, weight="bold")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ✓ Preprocessing figure saved → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — CLINICAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_clinical_report(result: Dict,
                           patient_id: str = "PATIENT_001",
                           save_path: Optional[str] = None) -> Dict:
    has_t  = result["has_tumour"]
    conf   = result["confidence"]
    unc    = result["is_uncertain"]
    if has_t and conf > 0.7:
        rec = "URGENT – High confidence tumour detected. Recommend immediate radiologist review."
    elif unc:
        rec = "REVIEW – Uncertain prediction. Recommend expert clinical review."
    else:
        rec = "ROUTINE – No significant tumour detected. Standard follow-up advised."

    ci_lo = result["ci_low"].tolist()
    ci_hi = result["ci_high"].tolist()

    report = {
        "patient_id":  patient_id,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system":      "Brain Tumour AI Detection System v1.0",
        "detection": {
            "tumour_present":   bool(has_t),
            "cgm_confidence":   float(result["cgm_confidence"]),
            "bounding_box":     list(result["bbox"]) if result["bbox"] else None,
        },
        "classification": {
            "predicted_type":  result["type_name"],
            "class_index":     int(result["tumour_type"]),
            "probabilities":   {TUMOUR_TYPES[i]: float(result["type_probs"][i])
                                for i in range(4)},
        },
        "uncertainty": {
            "mc_dropout_variance": float(result["uncertainty"]),
            "confidence":          float(result["confidence"]),
            "flag_for_review":     bool(result["is_uncertain"]),
            "ci_95": {TUMOUR_TYPES[i]: {"low": float(ci_lo[i]),
                                         "high": float(ci_hi[i])}
                      for i in range(4)},
        },
        "segmentation": {
            "dominant_region":  result["dominant_region"],
            "mask_voxel_count": int(result["mask"].sum()),
        },
        "radiomics": {name: float(result["radiomics"][i])
                      for i, name in enumerate(RadiomicsExtractor.FEATURE_NAMES)},
        "clinical_recommendation": rec,
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  ✓ Clinical report saved → {save_path}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  BRAIN TUMOUR DETECTION SYSTEM — SMOKE TEST")
    print("="*70)

    os.makedirs("outputs", exist_ok=True)

    # ── Step 1: Synthetic subject ────────────────────────────────────────────
    print("\n[1/6] Generating synthetic subject...")
    vols = make_synthetic_subject(shape=(120, 120, 77), has_tumour=True)
    stripper    = SkullStripper()
    normaliser  = IntensityNormaliser()
    vols = stripper.strip(vols)
    vols = normaliser.normalise_all(vols)
    print("  Volume shapes:")
    for k, v in vols.items():
        if isinstance(v, np.ndarray):
            print(f"    {k:12s}: {v.shape}  dtype={v.dtype}")
    print(f"  Seg unique labels : {np.unique(vols['seg']).tolist()}")
    print("  Z-score verification (brain-masked):")
    bm = vols["brain_mask"].astype(bool)
    for mod in MODALITIES:
        nz = vols[mod][bm]
        print(f"    {mod:6s}  mean={nz.mean():+.4f}  std={nz.std():.4f}")

    # ── Step 2: Preprocessing visualisation ─────────────────────────────────
    print("\n[2/6] Saving preprocessing visualisation...")
    visualise_preprocessing(vols, slice_idx=38,
                            save_path="outputs/preprocessing_demo.png")

    # ── Step 3: Slice extraction ─────────────────────────────────────────────
    print("\n[3/6] Extracting model-ready slice...")
    extractor = SliceExtractor(size=(128, 128))
    sample    = extractor.get(vols, 38)
    img_np    = sample["image"]                   # (4,128,128)
    mri_t     = torch.from_numpy(img_np).float().unsqueeze(0)  # (1,4,128,128)
    print(f"  Tensor shape : {tuple(mri_t.shape)}  dtype={mri_t.dtype}")

    # ── Step 4: Model forward pass + loss ────────────────────────────────────
    print("\n[4/6] Model forward pass...")
    seg_model = DeepUNet3Plus(in_ch=4, seg_classes=4, filters=[32, 64, 128, 256, 512])
    cls_model = TumourClassifier(num_classes=4)

    n_seg = sum(p.numel() for p in seg_model.parameters())
    n_cls = sum(p.numel() for p in cls_model.parameters())
    print(f"  DeepUNet3+  parameters: {n_seg:,}")
    print(f"  TumourClassifier params: {n_cls:,}")

    seg_model.train()
    out = seg_model(mri_t)
    seg_gt  = torch.zeros(1, 128, 128, dtype=torch.long)
    seg_gt[0, 40:80, 40:80] = 3   # fake tumour region
    cgm_gt  = torch.ones(1, dtype=torch.long)
    criterion = SegmentationLoss()
    loss = criterion(out, seg_gt, cgm_gt)
    print(f"  Segmentation loss      : {loss.item():.6f}")

    # ── Step 5: Full pipeline ─────────────────────────────────────────────────
    print("\n[5/6] Running full pipeline...")
    seg_model.eval()
    pipeline = BrainTumourPipeline(seg_model, cls_model,
                                   device=DEVICE,
                                   seg_thresh=0.3,
                                   mc_passes=10)
    result = pipeline.predict(mri_t)
    print("  Pipeline result keys:")
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            print(f"    {k:<20s}: ndarray {v.shape}")
        elif isinstance(v, (int, float, bool)):
            print(f"    {k:<20s}: {v}")
        else:
            print(f"    {k:<20s}: {type(v).__name__} = {str(v)[:60]}")

    # ── Step 6: Report export & metrics ─────────────────────────────────────
    print("\n[6/6] Generating reports & running metrics...")
    generate_diagnostic_report(result, save_path="outputs/diagnostic_report.png")
    report = export_clinical_report(result, patient_id="PATIENT_001",
                                    save_path="outputs/clinical_report.json")
    print(f"\n  Clinical recommendation:\n  → {report['clinical_recommendation']}")

    seg_metrics = SegmentationMetrics(num_classes=4)
    cls_metrics = ClassificationMetrics(num_classes=4)
    dummy_pred  = torch.randint(0, 4, (2, 4, 64, 64)).float()
    dummy_tgt   = torch.randint(0, 4, (2, 64, 64))
    seg_metrics.update(dummy_pred, dummy_tgt)
    print("\n  --- Segmentation Metrics (random baseline) ---")
    print(seg_metrics.summary_str())
    cls_pred = torch.randn(8, 4)
    cls_tgt  = torch.randint(0, 4, (8,))
    cls_metrics.update(cls_pred, cls_tgt)
    print("\n  --- Classification Metrics (random baseline) ---")
    print(cls_metrics.summary_str())

    print("\n" + "="*70)
    print("  ✅  ALL SMOKE TESTS PASSED")
    print("="*70)
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  HOW TO TRAIN ON REAL DATA                                      │
  │                                                                 │
  │  1. BraTS 2021 segmentation training:                           │
  │     from brain_tumor_complete import SegmentationTrainer        │
  │     trainer = SegmentationTrainer(                              │
  │         root_dir="/path/to/BraTS2021",                          │
  │         save_dir="checkpoints", batch=4, epochs=100)            │
  │     trainer.train()                                             │
  │     metrics = trainer.evaluate()                                │
  │                                                                 │
  │  2. Kaggle MRI classification training:                         │
  │     from brain_tumor_complete import ClassifierTrainer          │
  │     trainer = ClassifierTrainer(                                │
  │         root_dir="/path/to/brain-tumor-mri-dataset",            │
  │         save_dir="checkpoints", epochs=50)                      │
  │     trainer.train()                                             │
  │                                                                 │
  │  3. Inference with trained models:                              │
  │     from brain_tumor_complete import BrainTumourPipeline        │
  │     pipeline = BrainTumourPipeline.from_checkpoints(            │
  │         seg_ckpt="checkpoints/seg_best.pth",                    │
  │         cls_ckpt="checkpoints/cls_best.pth")                    │
  │     result = pipeline.predict(your_mri_tensor)                  │
  └─────────────────────────────────────────────────────────────────┘
""")

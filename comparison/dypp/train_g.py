from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
VIDEO_RE = re.compile(r"^S(?P<S>\d{3})C(?P<C>\d{3})P(?P<P>\d{3})R(?P<R>\d{3})A(?P<A>\d{3})$")
def parse_video_name(video_dir_name: str) -> Optional[Dict[str, int]]:
    m = VIDEO_RE.match(video_dir_name)
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}
def stable_int_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)
def load_ntu_splits(split_file: str) -> Dict[str, List[int]]:
    env: Dict[str, object] = {"range": range}
    with open(split_file, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, {"__builtins__": {}}, env)
    out: Dict[str, List[int]] = {}
    for k, v in env.items():
        if k.startswith("NTU") and isinstance(v, list):
            out[k] = [int(x) for x in v]
    return out
def is_in_split(meta: Dict[str, int], splits: Dict[str, List[int]], protocol: str, split: str) -> bool:
    protocol = protocol.lower()
    split = split.lower()
    if protocol == "ntu120_csub":
        train_ps = set(splits["NTU120_CSub_Train"])
        return (meta["P"] in train_ps) if split == "train" else (meta["P"] not in train_ps)
    if protocol == "ntu60_csub":
        train_ps = set(splits["NTU60_CSub_Train"])
        return (meta["P"] in train_ps) if split == "train" else (meta["P"] not in train_ps)
    if protocol == "ntu60_cview":
        train_cs = set(splits["NTU60_CView_Train"])
        return (meta["C"] in train_cs) if split == "train" else (meta["C"] not in train_cs)
    if protocol == "ntu120_cset":
        train_ss = set(splits["NTU120_CSet_Train"])
        val_ss = set(splits.get("NTU120_CSet_VAL", []))
        if split == "train":
            return meta["S"] in train_ss
        if split == "val":
            return meta["S"] in val_ss
        return (meta["S"] not in train_ss) and (meta["S"] not in val_ss)
    raise ValueError(f"Unknown protocol: {protocol}")
def list_frames(video_dir: str) -> List[str]:
    frames: List[str] = []
    for fn in os.listdir(video_dir):
        l = fn.lower()
        if l.endswith(".jpg") or l.endswith(".jpeg") or l.endswith(".png"):
            frames.append(fn)
    def _key(x: str) -> Tuple[int, str]:
        m = re.match(r"^(\d+)", x)
        return (int(m.group(1)) if m else 10**9, x)
    frames.sort(key=_key)
    return frames
class NTUFrameDataset(Dataset):
    def __init__(
        self,
        input_root: str,
        split_file: str,
        protocol: str,
        split: str,
        max_frames_per_video: int,
        seed: int,
    ):
        self.input_root = input_root
        self.protocol = protocol
        self.split = split
        self.max_frames_per_video = int(max_frames_per_video)
        self.seed = int(seed)
        splits = load_ntu_splits(split_file)
        all_videos = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
        all_videos.sort()
        rng = random.Random(self.seed)
        samples: List[Tuple[str, str, int, int]] = []
        max_p = 0
        for v in all_videos:
            meta = parse_video_name(v)
            if meta is None:
                continue
            if not is_in_split(meta, splits, protocol, split):
                continue
            in_dir = os.path.join(input_root, v)
            frames = list_frames(in_dir)
            if not frames:
                continue
            if self.max_frames_per_video > 0 and len(frames) > self.max_frames_per_video:
                rng.shuffle(frames)
                frames = sorted(frames[: self.max_frames_per_video])
            a = meta["A"] - 1
            p = meta["P"] - 1
            max_p = max(max_p, p)
            for fn in frames:
                samples.append((v, fn, a, p))
        self.samples = samples
        self.num_actions = 120
        self.num_persons = max_p + 1
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int):
        v, fn, a, p = self.samples[idx]
        path = os.path.join(self.input_root, v, fn)
        img = Image.open(path).convert("RGB")
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        return x, int(a), int(p), v, fn
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.flatten(1)
        return self.fc(h)
def _noll_to_nm(j: int) -> Tuple[int, int]:
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    j0 = n * (n + 1) // 2
    k = j - j0 - 1
    ms: List[int] = []
    if n % 2 == 0:
        ms.append(0)
        for a in range(2, n + 1, 2):
            ms.append(a)
            ms.append(-a)
    else:
        for a in range(1, n + 1, 2):
            ms.append(a)
            ms.append(-a)
    m = ms[k]
    return n, m
def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho, dtype=np.float32)
    out = np.zeros_like(rho, dtype=np.float32)
    half1 = (n + m) // 2
    half2 = (n - m) // 2
    for s in range(half2 + 1):
        num = math.factorial(n - s)
        den = (
            math.factorial(s)
            * math.factorial(half1 - s)
            * math.factorial(half2 - s)
        )
        c = ((-1) ** s) * (num / den)
        out += np.float32(c) * (rho ** (n - 2 * s)).astype(np.float32)
    return out
def zernike_noll(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n, m = _noll_to_nm(j)
    R = _zernike_radial(n, m, rho)
    if m == 0:
        Z = R
        norm = np.sqrt(np.float32(n + 1.0)).astype(np.float32)
        return (norm * Z).astype(np.float32)
    if m > 0:
        Z = R * np.cos(m * theta).astype(np.float32)
    else:
        Z = R * np.sin(abs(m) * theta).astype(np.float32)
    norm = np.sqrt(np.float32(2.0 * (n + 1.0))).astype(np.float32)
    return (norm * Z).astype(np.float32)
def unit_disk_polar_grid(hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = hw
    yy = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0) / ((h - 1) / 2.0 + 1e-8)
    xx = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0) / ((w - 1) / 2.0 + 1e-8)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    rho = np.sqrt(X * X + Y * Y).astype(np.float32)
    theta = np.arctan2(Y, X).astype(np.float32)
    mask = (rho <= 1.0).astype(np.float32)
    rho = np.clip(rho, 0.0, 1.0)
    return rho, theta, mask
def precompute_zernike_basis(dz: int, pupil_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    rho, theta, mask = unit_disk_polar_grid(pupil_hw)
    Hp, Wp = pupil_hw
    basis = np.zeros((dz, Hp, Wp), dtype=np.float32)
    for j in range(1, dz + 1):
        basis[j - 1] = zernike_noll(j, rho, theta) * mask
    return torch.from_numpy(basis), torch.from_numpy(mask)
class DyPPEmbeddingMLP(nn.Module):
    def __init__(self, dz: int = 350, hidden: int = 512, negative_slope: float = 0.2):
        super().__init__()
        self.dz = int(dz)
        self.hidden = int(hidden)
        self.negative_slope = float(negative_slope)
        self.fc1 = nn.Linear(self.dz, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, self.dz)
    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(eps), negative_slope=self.negative_slope, inplace=True)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope, inplace=True)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.negative_slope, inplace=True)
        return self.fc4(x)
    def save_npz(self, path: str) -> None:
        w = {
            "W1": self.fc1.weight.detach().cpu().numpy().T,
            "b1": self.fc1.bias.detach().cpu().numpy(),
            "W2": self.fc2.weight.detach().cpu().numpy().T,
            "b2": self.fc2.bias.detach().cpu().numpy(),
            "W3": self.fc3.weight.detach().cpu().numpy().T,
            "b3": self.fc3.bias.detach().cpu().numpy(),
            "W4": self.fc4.weight.detach().cpu().numpy().T,
            "b4": self.fc4.bias.detach().cpu().numpy(),
        }
        np.savez(path, **w)
@dataclass(frozen=True)
class CameraConfig:
    dz: int
    pupil_hw: int
    alpha_opd_scale_m: float
    noise_sigma: float
    rgb_wavelengths_nm: Tuple[float, float, float]
class DyPPTorchCamera(nn.Module):
    def __init__(self, cfg: CameraConfig, basis: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("basis", basis)
        self.register_buffer("mask", mask)
    @staticmethod
    def _center_pad_complex(src: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        B, Hs, Ws = src.shape
        Ho, Wo = out_hw
        if Hs > Ho or Ws > Wo:
            y0 = (Hs - Ho) // 2
            x0 = (Ws - Wo) // 2
            return src[:, y0:y0 + Ho, x0:x0 + Wo]
        out = torch.zeros((B, Ho, Wo), device=src.device, dtype=src.dtype)
        y0 = (Ho - Hs) // 2
        x0 = (Wo - Ws) // 2
        out[:, y0:y0 + Hs, x0:x0 + Ws] = src
        return out
    def synthesize_psf_rgb(self, alpha: torch.Tensor, out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, dz = alpha.shape
        assert dz == self.cfg.dz
        Hp = self.cfg.pupil_hw
        Wp = self.cfg.pupil_hw
        phi_dimless = torch.einsum("bd,dhw->bhw", alpha, self.basis)
        phi = phi_dimless * float(self.cfg.alpha_opd_scale_m)
        phi = phi * self.mask
        psfs: List[torch.Tensor] = []
        psf_ffts: List[torch.Tensor] = []
        for wl_nm in self.cfg.rgb_wavelengths_nm:
            lam = float(wl_nm) * 1e-9
            k = 2.0 * np.pi / lam
            t = torch.exp(1j * torch.tensor(k, device=phi.device, dtype=torch.float32) * phi.to(torch.float32))
            t = t.to(torch.complex64) * self.mask.to(torch.complex64)
            t_pad = self._center_pad_complex(t, out_hw)
            H = torch.fft.fft2(t_pad)
            psf = (H.abs() ** 2).to(torch.float32)
            psf = psf * (1.0 / (lam * lam))
            psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-12)
            psfs.append(psf)
            psf_ffts.append(torch.fft.rfft2(psf))
        psf_rgb = torch.stack(psfs, dim=1)
        psf_fft_rgb = torch.stack(psf_ffts, dim=1)
        return psf_rgb, psf_fft_rgb
    def forward(self, x_rgb: torch.Tensor, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x_rgb.shape
        psf_rgb, psf_fft_rgb = self.synthesize_psf_rgb(alpha, out_hw=(H, W))
        X = torch.fft.rfft2(x_rgb)
        Y = torch.fft.irfft2(X * psf_fft_rgb, s=(H, W))
        if self.cfg.noise_sigma > 0:
            Y = Y + torch.randn_like(Y) * float(self.cfg.noise_sigma)
        Y = torch.clamp(Y, 0.0, 1.0)
        return Y, psf_fft_rgb
def margin_sigmoid_accuracy(logits: torch.Tensor, y: torch.Tensor, eta: float) -> torch.Tensor:
    B, K = logits.shape
    y = y.long()
    logit_y = logits[torch.arange(B, device=logits.device), y]
    tmp = logits.clone()
    tmp[torch.arange(B, device=logits.device), y] = -1e9
    max_other, _ = tmp.max(dim=1)
    margin = logit_y - max_other
    return torch.sigmoid(margin * float(eta)).mean()
def l_noninvert(psf_fft_rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mag2 = (psf_fft_rgb.real ** 2 + psf_fft_rgb.imag ** 2)
    return (-(eps / (mag2 + eps))).mean()
def l_diversity(alpha_samples: torch.Tensor, mode: str = "diag", eps: float = 1e-3) -> torch.Tensor:
    A = alpha_samples
    A = A - A.mean(dim=0, keepdim=True)
    if mode == "diag":
        var = (A ** 2).mean(dim=0)
        return -var.mean()
    if mode == "logdet":
        N, dz = A.shape
        cov = (A.t() @ A) / max(N - 1, 1)
        cov = cov + torch.eye(dz, device=A.device, dtype=A.dtype) * float(eps)
        sign, logabsdet = torch.linalg.slogdet(cov)
        return -(0.5 * logabsdet) + (sign <= 0).to(A.dtype) * 10.0
    raise ValueError(f"Unknown diversity mode: {mode}")
def freeze_params(m: nn.Module, freeze: bool) -> None:
    for p in m.parameters():
        p.requires_grad = not freeze
def main() -> None:
    ap = argparse.ArgumentParser(description="Train DyPP g on NTU for action utility & person privacy.")
    ap.add_argument("--input_root", type=str, default="/data2/NTU_data/resize_all_gt")
    ap.add_argument("--split_file", type=str, default="/data2/NTU_data/a_TIP_rebuttel/DYPP/NTURGBD_split.txt")
    ap.add_argument("--protocol", type=str, default="ntu120_csub", choices=["ntu120_csub", "ntu120_cset", "ntu60_csub", "ntu60_cview"])
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--max_frames_per_video", type=int, default=16, help="subsample frames per video to control dataset size (0 means all).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dz", type=int, default=350)
    ap.add_argument("--pupil_hw", type=int, default=64, help="pupil grid size (Hp=Wp). Paper uses larger; this is a compute-friendly default.")
    ap.add_argument("--alpha_opd_scale_m", type=float, default=1e-6, help="OPD scale in meters (paper not specified).")
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_action", type=float, default=1e-4)
    ap.add_argument("--lr_person", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--privacy_p", type=float, default=0.1, help="target upper bound on person-ID recognizer accuracy proxy.")
    ap.add_argument("--eta_margin", type=float, default=10.0, help="sigmoid smoothness for margin.")
    ap.add_argument("--lambda_noninvert", type=float, default=1.0)
    ap.add_argument("--lambda_diversity", type=float, default=0.1)
    ap.add_argument("--lambda_utility", type=float, default=1.0)
    ap.add_argument("--lambda_privacy", type=float, default=1.0)
    ap.add_argument("--diversity_mode", type=str, default="diag", choices=["diag", "logdet"])
    ap.add_argument("--ncov", type=int, default=64, help="number of alpha samples for diversity loss per step")
    ap.add_argument("--save_dir", type=str, default="/data2/NTU_data/a_TIP_rebuttel/DYPP/checkpoints")
    ap.add_argument("--save_every", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = NTUFrameDataset(
        input_root=args.input_root,
        split_file=args.split_file,
        protocol=args.protocol,
        split=args.split,
        max_frames_per_video=args.max_frames_per_video,
        seed=args.seed,
    )
    if len(ds) == 0:
        raise RuntimeError("Empty dataset. Check input_root/protocol/split.")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    g = DyPPEmbeddingMLP(dz=args.dz, hidden=args.hidden).to(device)
    action_net = SmallCNN(num_classes=ds.num_actions).to(device)
    person_net = SmallCNN(num_classes=ds.num_persons).to(device)
    basis, mask = precompute_zernike_basis(dz=args.dz, pupil_hw=(args.pupil_hw, args.pupil_hw))
    basis = basis.to(device)
    mask = mask.to(device)
    cam_cfg = CameraConfig(
        dz=args.dz,
        pupil_hw=args.pupil_hw,
        alpha_opd_scale_m=float(args.alpha_opd_scale_m),
        noise_sigma=float(args.noise_sigma),
        rgb_wavelengths_nm=(640.0, 550.0, 460.0),
    )
    cam = DyPPTorchCamera(cfg=cam_cfg, basis=basis, mask=mask).to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr_g)
    opt_a = torch.optim.Adam(action_net.parameters(), lr=args.lr_action)
    opt_p = torch.optim.Adam(person_net.parameters(), lr=args.lr_person)
    step = 0
    for epoch in range(int(args.epochs)):
        it = iter(dl)
        for _ in range(int(args.steps_per_epoch)):
            try:
                x, y_action, y_person, vname, fname = next(it)
            except StopIteration:
                it = iter(dl)
                x, y_action, y_person, vname, fname = next(it)
            x = x.to(device, non_blocking=True)
            y_action = y_action.to(device, non_blocking=True)
            y_person = y_person.to(device, non_blocking=True)
            B = x.shape[0]
            freeze_params(g, True)
            freeze_params(action_net, False)
            freeze_params(person_net, True)
            action_net.train()
            person_net.eval()
            with torch.no_grad():
                eps = (torch.rand((B, args.dz), device=device) * 2.0 - 1.0)
                alpha = g(eps)
                x_blur, _ = cam(x, alpha)
            logits_a = action_net(x_blur)
            loss_a = F.cross_entropy(logits_a, y_action)
            opt_a.zero_grad(set_to_none=True)
            loss_a.backward()
            opt_a.step()
            freeze_params(person_net, False)
            person_net.train()
            with torch.no_grad():
                eps = (torch.rand((B, args.dz), device=device) * 2.0 - 1.0)
                alpha = g(eps)
                x_blur, _ = cam(x, alpha)
            logits_p = person_net(x_blur)
            loss_p = F.cross_entropy(logits_p, y_person)
            opt_p.zero_grad(set_to_none=True)
            loss_p.backward()
            opt_p.step()
            freeze_params(g, False)
            freeze_params(action_net, True)
            freeze_params(person_net, True)
            g.train()
            action_net.eval()
            person_net.eval()
            eps = (torch.rand((B, args.dz), device=device) * 2.0 - 1.0)
            alpha = g(eps)
            x_blur, psf_fft_rgb = cam(x, alpha)
            logits_a = action_net(x_blur)
            logits_p = person_net(x_blur)
            Lutil = F.cross_entropy(logits_a, y_action)
            acc_p_soft = margin_sigmoid_accuracy(logits_p, y_person, eta=float(args.eta_margin))
            Lpriv = F.relu(acc_p_soft - float(args.privacy_p))
            Lni = l_noninvert(psf_fft_rgb, eps=1e-6)
            eps_cov = (torch.rand((int(args.ncov), args.dz), device=device) * 2.0 - 1.0)
            alpha_cov = g(eps_cov)
            Ldiv = l_diversity(alpha_cov, mode=args.diversity_mode, eps=1e-3)
            loss_g = (
                float(args.lambda_utility) * Lutil
                + float(args.lambda_privacy) * Lpriv
                + float(args.lambda_noninvert) * Lni
                + float(args.lambda_diversity) * Ldiv
            )
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            step += 1
            if step % 50 == 0:
                with torch.no_grad():
                    top1_a = (logits_a.argmax(dim=1) == y_action).float().mean().item()
                    top1_p = (logits_p.argmax(dim=1) == y_person).float().mean().item()
                print(
                    f"step={step} "
                    f"Lg={loss_g.item():.4f} Lutil={Lutil.item():.4f} Lpriv={Lpriv.item():.4f} "
                    f"Lni={Lni.item():.4f} Ldiv={Ldiv.item():.4f} "
                    f"accA={top1_a:.3f} accP={top1_p:.3f} accP_soft={acc_p_soft.item():.3f}"
                )
            if step % int(args.save_every) == 0:
                ckpt_npz = os.path.join(args.save_dir, f"g_step{step}.npz")
                g.save_npz(ckpt_npz)
                meta = {
                    "dz": int(args.dz),
                    "hidden": int(args.hidden),
                    "pupil_hw": int(args.pupil_hw),
                    "alpha_opd_scale_m": float(args.alpha_opd_scale_m),
                    "protocol": args.protocol,
                    "split": args.split,
                    "max_frames_per_video": int(args.max_frames_per_video),
                    "privacy_p": float(args.privacy_p),
                    "diversity_mode": args.diversity_mode,
                }
                with open(os.path.join(args.save_dir, f"g_step{step}.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE] {ckpt_npz}")
    ckpt_npz = os.path.join(args.save_dir, "g_final.npz")
    g.save_npz(ckpt_npz)
    print(f"[DONE] saved g weights: {ckpt_npz}")
if __name__ == "__main__":
    main()

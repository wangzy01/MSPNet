from __future__ import annotations
import argparse
import hashlib
import math
import os
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
try:
    import scipy.fft as fft_module
except ImportError:
    import numpy.fft as fft_module
VIDEO_RE = re.compile(r"^S(?P<S>\d{3})C(?P<C>\d{3})P(?P<P>\d{3})R(?P<R>\d{3})A(?P<A>\d{3})$")
def to_float01_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.ascontiguousarray(np.asarray(img), dtype=np.float32) / 255.0
def save_image(path: str, img01: np.ndarray) -> None:
    img01 = np.clip(img01, 0.0, 1.0)
    Image.fromarray((img01 * 255.0).astype(np.uint8), mode="RGB").save(path)
def normalize_01_inplace(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = float(x.min())
    mx = float(x.max())
    x -= mn
    x *= 1.0 / (mx - mn + eps)
    return x
def stable_int_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)
def _noll_to_nm(j: int) -> Tuple[int, int]:
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    j0 = n * (n + 1) // 2
    k = j - j0 - 1
    ms = []
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
def _unit_disk_polar_grid(hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = hw
    yy = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0) / ((h - 1) / 2.0 + 1e-8)
    xx = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0) / ((w - 1) / 2.0 + 1e-8)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    rho = np.sqrt(X * X + Y * Y).astype(np.float32)
    theta = np.arctan2(Y, X).astype(np.float32)
    mask = (rho <= 1.0).astype(np.float32)
    rho = np.clip(rho, 0.0, 1.0)
    return rho, theta, mask
@dataclass
class DyPPMLPConfig:
    dz: int = 350
    hidden: int = 512
    negative_slope: float = 0.2
class DyPPEmbeddingMLP:
    def __init__(self, cfg: DyPPMLPConfig = DyPPMLPConfig(), seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        dz = cfg.dz
        h = cfg.hidden
        self.W1 = (rng.standard_normal((dz, h)).astype(np.float32) * np.float32(0.02))
        self.b1 = np.zeros((h,), dtype=np.float32)
        self.W2 = (rng.standard_normal((h, h)).astype(np.float32) * np.float32(0.02))
        self.b2 = np.zeros((h,), dtype=np.float32)
        self.W3 = (rng.standard_normal((h, h)).astype(np.float32) * np.float32(0.02))
        self.b3 = np.zeros((h,), dtype=np.float32)
        self.W4 = (rng.standard_normal((h, dz)).astype(np.float32) * np.float32(0.02))
        self.b4 = np.zeros((dz,), dtype=np.float32)
    def load_npz(self, weights_npz_path: str) -> None:
        data = np.load(weights_npz_path)
        for k in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"):
            if k not in data:
                raise ValueError(f"Missing key '{k}' in weights file: {weights_npz_path}")
        self.W1 = data["W1"].astype(np.float32)
        self.b1 = data["b1"].astype(np.float32)
        self.W2 = data["W2"].astype(np.float32)
        self.b2 = data["b2"].astype(np.float32)
        self.W3 = data["W3"].astype(np.float32)
        self.b3 = data["b3"].astype(np.float32)
        self.W4 = data["W4"].astype(np.float32)
        self.b4 = data["b4"].astype(np.float32)
        dz = self.cfg.dz
        h = self.cfg.hidden
        expected = {
            "W1": (dz, h), "b1": (h,),
            "W2": (h, h), "b2": (h,),
            "W3": (h, h), "b3": (h,),
            "W4": (h, dz), "b4": (dz,),
        }
        for k, shp in expected.items():
            if getattr(self, k).shape != shp:
                raise ValueError(f"Bad shape for {k}: got {getattr(self, k).shape}, expected {shp}")
    def _lrelu(self, x: np.ndarray) -> np.ndarray:
        a = np.float32(self.cfg.negative_slope)
        return np.where(x >= 0, x, a * x).astype(np.float32)
    def forward(self, eps: np.ndarray) -> np.ndarray:
        if eps.shape != (self.cfg.dz,):
            raise ValueError(f"eps shape must be ({self.cfg.dz},) but got {eps.shape}")
        x = eps.astype(np.float32)
        x = self._lrelu(x @ self.W1 + self.b1)
        x = self._lrelu(x @ self.W2 + self.b2)
        x = self._lrelu(x @ self.W3 + self.b3)
        alpha = x @ self.W4 + self.b4
        return alpha.astype(np.float32)
@dataclass(frozen=True)
class DyPPCameraConfig:
    dz: int = 350
    rgb_wavelengths_nm: Tuple[float, float, float] = (640.0, 550.0, 460.0)
    pixel_size_um: float = 1.0
    f_number: float = 1.8
    pupil_hw: Tuple[int, int] = (256, 256)
    alpha_opd_scale_m: float = 1e-6
    noise_sigma: float = 0.0
    alpha_scale: float = 1.0
    normalize_output: bool = True
    eps: float = 1e-8
class DyPPCamera:
    def __init__(
        self,
        cam_cfg: DyPPCameraConfig = DyPPCameraConfig(),
        mlp: Optional[DyPPEmbeddingMLP] = None,
        mlp_cfg: Optional[DyPPMLPConfig] = None,
        mlp_seed: int = 0,
    ):
        self.cfg = cam_cfg
        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = DyPPEmbeddingMLP(cfg=mlp_cfg or DyPPMLPConfig(dz=cam_cfg.dz), seed=mlp_seed)
        rho, theta, mask = _unit_disk_polar_grid(cam_cfg.pupil_hw)
        self._rho = rho
        self._theta = theta
        self._mask = mask
        self._zernike_basis = self._precompute_zernike_basis(dz=cam_cfg.dz)
    def _precompute_zernike_basis(self, dz: int) -> np.ndarray:
        Hp, Wp = self.cfg.pupil_hw
        basis = np.zeros((dz, Hp, Wp), dtype=np.float32)
        for j in range(1, dz + 1):
            Z = zernike_noll(j, self._rho, self._theta)
            basis[j - 1] = (Z * self._mask).astype(np.float32)
        return basis
    def sample_eps(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(self.cfg.dz,)).astype(np.float32)
    def sample_alpha(self, rng: np.random.Generator) -> np.ndarray:
        eps = self.sample_eps(rng)
        alpha = self.mlp.forward(eps)
        alpha = alpha * np.float32(self.cfg.alpha_scale)
        return alpha.astype(np.float32)
    def _pupil_phase_from_alpha(self, alpha: np.ndarray) -> np.ndarray:
        phi_dimless = np.tensordot(alpha.astype(np.float32), self._zernike_basis, axes=(0, 0)).astype(np.float32)
        opd_m = phi_dimless * np.float32(self.cfg.alpha_opd_scale_m)
        return (opd_m * self._mask).astype(np.float32)
    @staticmethod
    def _center_pad_complex(src: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
        Hs, Ws = src.shape
        Ho, Wo = out_hw
        if Hs > Ho or Ws > Wo:
             y0 = (Hs - Ho) // 2
             x0 = (Ws - Wo) // 2
             return src[y0:y0+Ho, x0:x0+Wo]
        out = np.zeros((Ho, Wo), dtype=np.complex64)
        y0 = (Ho - Hs) // 2
        x0 = (Wo - Ws) // 2
        out[y0:y0 + Hs, x0:x0 + Ws] = src.astype(np.complex64)
        return out
    def synthesize_psf(self, alpha: np.ndarray, out_hw: Tuple[int, int], wavelength_nm: float) -> np.ndarray:
        lam_m = float(wavelength_nm) * 1e-9
        k = (2.0 * np.pi / lam_m)
        phi = self._pupil_phase_from_alpha(alpha)
        t = (np.exp(1j * np.float32(k) * phi).astype(np.complex64) * self._mask.astype(np.complex64))
        t_pad = self._center_pad_complex(t, out_hw)
        H = fft_module.fft2(t_pad)
        psf = (np.abs(H) ** 2).astype(np.float32)
        psf *= np.float32(1.0 / (lam_m * lam_m))
        s = float(psf.sum())
        if s > 0:
            psf /= np.float32(s)
        return np.ascontiguousarray(psf.astype(np.float32))
    def _convolve_channel_circular(self, img_ch: np.ndarray, psf: np.ndarray) -> np.ndarray:
        img_fft = fft_module.rfft2(img_ch.astype(np.float32))
        psf_fft = fft_module.rfft2(psf.astype(np.float32))
        out = fft_module.irfft2(img_fft * psf_fft, s=img_ch.shape)
        return out.astype(np.float32)
    def capture(self, rgb: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
        rgb = rgb.astype(np.float32)
        h, w = rgb.shape[:2]
        rng = np.random.default_rng(seed)
        alpha = self.sample_alpha(rng)
        out = np.zeros((h, w, 3), dtype=np.float32)
        for c, wl in enumerate(self.cfg.rgb_wavelengths_nm):
            psf = self.synthesize_psf(alpha=alpha, out_hw=(h, w), wavelength_nm=wl)
            out[..., c] = self._convolve_channel_circular(rgb[..., c], psf)
        if self.cfg.noise_sigma > 0:
            out += rng.normal(0.0, self.cfg.noise_sigma, size=out.shape).astype(np.float32)
        out = np.clip(out, 0.0, 1.0)
        if self.cfg.normalize_output:
            normalize_01_inplace(out, eps=self.cfg.eps)
        return out
_CAM: Optional[DyPPCamera] = None
def _list_video_dirs(input_root: str) -> List[str]:
    out: List[str] = []
    for name in os.listdir(input_root):
        p = os.path.join(input_root, name)
        if os.path.isdir(p):
            out.append(name)
    out.sort()
    return out
def _list_frames(video_dir: str) -> List[str]:
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
def _init_worker(
    g_weights_npz: str,
    pupil_hw: int,
    alpha_opd_scale_m: float,
    noise_sigma: float,
    mlp_seed: int,
) -> None:
    global _CAM
    cam_cfg = DyPPCameraConfig(
        dz=350,
        rgb_wavelengths_nm=(640.0, 550.0, 460.0),
        pixel_size_um=1.0,
        f_number=1.8,
        pupil_hw=(int(pupil_hw), int(pupil_hw)),
        noise_sigma=float(noise_sigma),
        alpha_opd_scale_m=float(alpha_opd_scale_m),
    )
    cam = DyPPCamera(cam_cfg=cam_cfg, mlp_seed=int(mlp_seed))
    if g_weights_npz:
        cam.mlp.load_npz(g_weights_npz)
    _CAM = cam
def _process_one_video(args: Tuple[str, str, str, bool, bool]) -> Tuple[str, int, int, int]:
    global _CAM
    if _CAM is None:
        raise RuntimeError("Worker camera not initialized. This should not happen.")
    video_name, input_root, output_root, overwrite, skip_non_ntu = args
    if skip_non_ntu and (VIDEO_RE.match(video_name) is None):
        return (video_name, 0, 0, 0)
    in_dir = os.path.join(input_root, video_name)
    out_dir = os.path.join(output_root, video_name)
    os.makedirs(out_dir, exist_ok=True)
    frames = _list_frames(in_dir)
    n_failed = 0
    n_written = 0
    for fn in frames:
        in_path = os.path.join(in_dir, fn)
        out_path = os.path.join(out_dir, fn)
        if (not overwrite) and os.path.exists(out_path):
            continue
        try:
            img = Image.open(in_path)
            rgb = to_float01_rgb(img)
            seed = stable_int_seed(video_name, fn)
            out = _CAM.capture(rgb, seed=seed)
            save_image(out_path, out)
            n_written += 1
        except Exception as e:
            n_failed += 1
            print(f"[WARN] {in_path} failed: {e}")
    return (video_name, len(frames), n_written, n_failed)
def process_ntu_root(
    input_root: str,
    output_root: str,
    g_weights_npz: str,
    workers: int = 8,
    overwrite: bool = False,
    skip_non_ntu: bool = True,
    pupil_hw: int = 64,
    alpha_opd_scale_m: float = 1e-6,
    noise_sigma: float = 0.0,
    mlp_seed: int = 42,
) -> None:
    os.makedirs(output_root, exist_ok=True)
    videos = _list_video_dirs(input_root)
    if not videos:
        raise RuntimeError(f"No subdirectories found under {input_root}")
    print(f"Found {len(videos)} video dirs under {input_root}")
    print(f"Output root: {output_root}")
    print(f"Using g weights: {g_weights_npz}")
    print(f"pupil_hw={pupil_hw}, alpha_opd_scale_m={alpha_opd_scale_m}, noise_sigma={noise_sigma}, workers={workers}")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_worker,
        initargs=(g_weights_npz, int(pupil_hw), float(alpha_opd_scale_m), float(noise_sigma), int(mlp_seed)),
    ) as ex:
        futs = []
        for v in videos:
            futs.append(ex.submit(_process_one_video, (v, input_root, output_root, bool(overwrite), bool(skip_non_ntu))))
        done = 0
        total_frames = 0
        total_written = 0
        total_failed = 0
        for fut in as_completed(futs):
            video_name, n_frames, n_written, n_failed = fut.result()
            done += 1
            total_frames += n_frames
            total_written += n_written
            total_failed += n_failed
            if done % 200 == 0:
                print(f"[{done}/{len(videos)}] frames={total_frames} written={total_written} failed={total_failed}")
    print(f"[DONE] videos={len(videos)} frames={total_frames} written={total_written} failed={total_failed}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyPP camera forward simulation on NTU-style dataset (keep folder/file structure).")
    parser.add_argument("--input_root", type=str, default="/data2/NTU_data/resize_all_gt")
    parser.add_argument("--output_root", type=str, default="/data2/NTU_data/resize_all_gt_dypp_final")
    parser.add_argument(
        "--g_weights_npz",
        type=str,
        default="",
        help="Path to trained embedding network g weights (.npz with W1,b1,...).",
    )
    parser.add_argument("--workers", type=int, default=16, help="Process-level parallelism.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--skip_non_ntu", action="store_true", help="Skip non-NTU-format subfolders (default: False).")
    parser.add_argument("--pupil_hw", type=int, default=64, help="pupil grid size (match g_step1000.json by default).")
    parser.add_argument(
        "--alpha_opd_scale_m",
        type=float,
        default=1e-6,
        help="OPD scale in meters for Eq.(4) exp[i k φ]. Paper does not specify; set to match training if known.",
    )
    args = parser.parse_args()
    process_ntu_root(
        input_root=args.input_root,
        output_root=args.output_root,
        g_weights_npz=args.g_weights_npz,
        workers=int(args.workers),
        overwrite=bool(args.overwrite),
        skip_non_ntu=bool(args.skip_non_ntu),
        pupil_hw=int(args.pupil_hw),
        alpha_opd_scale_m=float(args.alpha_opd_scale_m),
        noise_sigma=0.0,
        mlp_seed=42,
    )

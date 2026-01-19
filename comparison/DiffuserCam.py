from __future__ import annotations
import os
import glob
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
from PIL import Image
@dataclass(frozen=True)
class DiffuserCamConfig:
    pixel_pitch_um: float = 6.5
    diffuser_sensor_distance_mm: float = 8.9
    diffuser_feature_um: float = 140.0
    diffuser_slope_deg: float = 0.7
    aperture_hw_mm: Tuple[float, float] = (5.5, 7.5)
    binning: int = 2
    z_min_mm: float = 10.86
    z_max_mm: float = 36.26
    n_depths: int = 128
    wavelength_nm: float = 532.0
    seed: int = 42
    eps: float = 1e-10
def to_float01_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0
def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
def binning_2x2(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    img = img[:h2, :w2]
    if img.ndim == 2:
        return img.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3)).astype(np.float32)
    else:
        c = img.shape[2]
        return img.reshape(h2 // factor, factor, w2 // factor, factor, c).mean(axis=(1, 3)).astype(np.float32)
def center_crop(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    h, w = img.shape[:2]
    if out_h > h or out_w > w:
        raise ValueError(f"Crop target size {out_hw} larger than input size {img.shape[:2]}")
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return img[y0:y0 + out_h, x0:x0 + out_w]
def fft_convolve2d_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, wa = a.shape
    hb, wb = b.shape
    out_h = ha + hb - 1
    out_w = wa + wb - 1
    fa = np.fft.rfft2(a, s=(out_h, out_w))
    fb = np.fft.rfft2(b, s=(out_h, out_w))
    out = np.fft.irfft2(fa * fb, s=(out_h, out_w))
    return out.astype(np.float32)
def fft_convolve2d_crop(a: np.ndarray, b: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    full = fft_convolve2d_full(a, b)
    return center_crop(full, out_hw)
def normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return ((x - mn) / (mx - mn + eps)).astype(np.float32)
def save_image(path: str, img: np.ndarray) -> None:
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        Image.fromarray((img * 255).astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(path)
class DiffuserCam:
    def __init__(self, cfg: DiffuserCamConfig = DiffuserCamConfig()):
        self.cfg = cfg
        self._phase_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._psf_cache: Dict[Tuple[int, int, float, float], np.ndarray] = {}
    def _make_coordinates(self, hw: Tuple[int, int], dx: float) -> Tuple[np.ndarray, np.ndarray]:
        h, w = hw
        y = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0) * dx
        x = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0) * dx
        Y, X = np.meshgrid(y, x, indexing="ij")
        return X, Y
    def _gaussian_lowpass(self, noise: np.ndarray, sigma_px: float) -> np.ndarray:
        h, w = noise.shape
        fy = np.fft.fftfreq(h).astype(np.float32)
        fx = np.fft.fftfreq(w).astype(np.float32)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        G = np.exp(-2.0 * (np.pi ** 2) * (sigma_px ** 2) * (FX ** 2 + FY ** 2))
        F = np.fft.fft2(noise)
        out = np.fft.ifft2(F * G).real.astype(np.float32)
        out -= out.mean()
        std = out.std()
        if std > 0:
            out /= std
        return out
    def get_diffuser_phase(self, hw: Tuple[int, int]) -> np.ndarray:
        if hw in self._phase_cache:
            return self._phase_cache[hw]
        rng = np.random.default_rng(self.cfg.seed)
        noise = rng.standard_normal(hw, dtype=np.float32)
        feature_px = self.cfg.diffuser_feature_um / self.cfg.pixel_pitch_um
        sigma_px = max(1.0, feature_px / 2.0)
        smooth = self._gaussian_lowpass(noise, sigma_px)
        phi = np.pi * np.tanh(smooth).astype(np.float32)
        self._phase_cache[hw] = phi
        return phi
    def _aperture_mask(self, hw: Tuple[int, int]) -> np.ndarray:
        if self.cfg.aperture_hw_mm is None:
            return np.ones(hw, dtype=np.float32)
        ap_h_mm, ap_w_mm = self.cfg.aperture_hw_mm
        dx_m = self.cfg.pixel_pitch_um * 1e-6
        ap_h_px = min(int(round(ap_h_mm * 1e-3 / dx_m)), hw[0])
        ap_w_px = min(int(round(ap_w_mm * 1e-3 / dx_m)), hw[1])
        mask = np.zeros(hw, dtype=np.float32)
        y0 = (hw[0] - ap_h_px) // 2
        x0 = (hw[1] - ap_w_px) // 2
        mask[y0:y0 + ap_h_px, x0:x0 + ap_w_px] = 1.0
        return mask
    def _angular_spectrum_propagate(
        self, 
        field: np.ndarray, 
        wavelength_m: float, 
        distance_m: float
    ) -> np.ndarray:
        h, w = field.shape
        dx = self.cfg.pixel_pitch_um * 1e-6
        k = 2.0 * np.pi / wavelength_m
        fx = np.fft.fftfreq(w, d=dx).astype(np.float32)
        fy = np.fft.fftfreq(h, d=dx).astype(np.float32)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        lam_fx = wavelength_m * FX
        lam_fy = wavelength_m * FY
        inside = 1.0 - lam_fx ** 2 - lam_fy ** 2
        sqrt_term = np.sqrt(np.maximum(inside, 0.0)).astype(np.float32)
        H = np.exp(1j * k * distance_m * sqrt_term).astype(np.complex64)
        H = np.where(inside >= 0, H, 0.0 + 0.0j)
        F = np.fft.fft2(field)
        return np.fft.ifft2(F * H)
    def generate_psf(
        self,
        hw: Tuple[int, int],
        z_mm: float,
        wavelength_nm: Optional[float] = None,
        normalize: bool = True
    ) -> np.ndarray:
        if wavelength_nm is None:
            wavelength_nm = self.cfg.wavelength_nm
        key = (hw[0], hw[1], float(z_mm), float(wavelength_nm))
        if key in self._psf_cache:
            return self._psf_cache[key]
        dx_m = self.cfg.pixel_pitch_um * 1e-6
        lam = wavelength_nm * 1e-9
        d_m = self.cfg.diffuser_sensor_distance_mm * 1e-3
        z_m = z_mm * 1e-3
        k = 2.0 * np.pi / lam
        phi = self.get_diffuser_phase(hw)
        aperture = self._aperture_mask(hw)
        X, Y = self._make_coordinates(hw, dx_m)
        quad_phase = np.exp(1j * (k / (2.0 * (z_m + self.cfg.eps))) * (X ** 2 + Y ** 2)).astype(np.complex64)
        field_at_diffuser = aperture * quad_phase * np.exp(1j * phi).astype(np.complex64)
        field_at_sensor = self._angular_spectrum_propagate(field_at_diffuser, lam, d_m)
        psf = np.abs(field_at_sensor) ** 2
        psf = np.maximum(psf, 0.0).astype(np.float32)
        if normalize:
            s = psf.sum()
            if s > 0:
                psf /= s
        self._psf_cache[key] = psf
        return psf
    def capture(
        self,
        image: np.ndarray,
        z_mm: float = 20.0,
        mode: str = "mono",
        do_binning: bool = True,
        normalize: bool = True,
        rgb_wavelengths_nm: Tuple[float, float, float] = (630.0, 530.0, 460.0)
    ) -> Tuple[np.ndarray, Dict]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be (H, W, 3) RGB")
        h, w = image.shape[:2]
        out_hw = (h, w)
        debug = {}
        if mode.lower() == "mono":
            luma = rgb_to_luma(image).astype(np.float32)
            psf = self.generate_psf(out_hw, z_mm)
            measurement = fft_convolve2d_crop(luma, psf, out_hw)
            debug["luma"] = luma
            debug["psf"] = psf
        elif mode.lower() == "rgb":
            measurement = np.zeros_like(image)
            psfs = []
            for c, wl in enumerate(rgb_wavelengths_nm):
                psf = self.generate_psf(out_hw, z_mm, wavelength_nm=wl)
                measurement[..., c] = fft_convolve2d_crop(image[..., c], psf, out_hw)
                psfs.append(psf)
            debug["psf_r"] = psfs[0]
            debug["psf_g"] = psfs[1]
            debug["psf_b"] = psfs[2]
        else:
            raise ValueError('mode must be "mono" or "rgb"')
        if do_binning:
            measurement = binning_2x2(measurement, self.cfg.binning)
        if normalize:
            measurement = normalize_01(measurement)
        return measurement.astype(np.float32), debug
    def capture_volume(
        self,
        volume: np.ndarray,
        z_planes_mm: np.ndarray,
        do_binning: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        if volume.ndim != 3:
            raise ValueError("volume must be (Nz, H, W) 3D array")
        if len(z_planes_mm) != volume.shape[0]:
            raise ValueError("z_planes_mm length must equal Nz")
        nz, h, w = volume.shape
        out_hw = (h, w)
        y = np.zeros(out_hw, dtype=np.float32)
        for i in range(nz):
            psf = self.generate_psf(out_hw, float(z_planes_mm[i]))
            y += fft_convolve2d_crop(volume[i].astype(np.float32), psf, out_hw)
        if do_binning:
            y = binning_2x2(y, self.cfg.binning)
        if normalize:
            y = normalize_01(y)
        return y.astype(np.float32), {"z_planes_mm": z_planes_mm}
    def reconstruct_2d_admm(
        self,
        measurement: np.ndarray,
        psf: np.ndarray,
        lambda_tv: float = 0.02,
        n_iterations: int = 200,
        mu1: float = 1.0,
        mu2: float = 1.0,
        mu3: float = 1.0,
        nonneg: bool = True,
        verbose: bool = False
    ) -> np.ndarray:
        if measurement.ndim != 2 or psf.ndim != 2:
            raise ValueError("measurement and psf must be 2D arrays")
        h, w = measurement.shape
        if psf.shape != (h, w):
            psf = center_crop(psf, (h, w)) if all(p >= s for p, s in zip(psf.shape, (h, w))) else psf
            if psf.shape != (h, w):
                psf_padded = np.zeros((h, w), dtype=np.float32)
                ph, pw = psf.shape
                y0, x0 = (h - ph) // 2, (w - pw) // 2
                psf_padded[y0:y0+ph, x0:x0+pw] = psf
                psf = psf_padded
        y = measurement.astype(np.float32)
        x = np.maximum(y.copy(), 0.0)
        Hf = np.fft.rfft2(psf)
        Hf_conj = np.conj(Hf)
        HtH = (Hf_conj * Hf).real.astype(np.float32)
        dx_kernel = np.zeros((h, w), dtype=np.float32)
        dy_kernel = np.zeros((h, w), dtype=np.float32)
        dx_kernel[0, 0] = -1.0
        dx_kernel[0, 1 % w] = 1.0
        dy_kernel[0, 0] = -1.0
        dy_kernel[1 % h, 0] = 1.0
        Dx_f = np.fft.rfft2(dx_kernel)
        Dy_f = np.fft.rfft2(dy_kernel)
        DtD = (np.conj(Dx_f) * Dx_f + np.conj(Dy_f) * Dy_f).real.astype(np.float32)
        ux = np.zeros((h, w), dtype=np.float32)
        uy = np.zeros((h, w), dtype=np.float32)
        v = np.zeros((h, w), dtype=np.float32)
        w_var = np.maximum(x, 0.0)
        xi = np.zeros((h, w), dtype=np.float32)
        eta_x = np.zeros((h, w), dtype=np.float32)
        eta_y = np.zeros((h, w), dtype=np.float32)
        rho = np.zeros((h, w), dtype=np.float32)
        denom = (mu1 * HtH + mu2 * DtD + mu3).astype(np.float32)
        def soft_threshold(a: np.ndarray, t: float) -> np.ndarray:
            return np.sign(a) * np.maximum(np.abs(a) - t, 0.0)
        def H_forward(img: np.ndarray) -> np.ndarray:
            return np.fft.irfft2(np.fft.rfft2(img) * Hf, s=(h, w)).real.astype(np.float32)
        def H_adjoint(img: np.ndarray) -> np.ndarray:
            return np.fft.irfft2(np.fft.rfft2(img) * Hf_conj, s=(h, w)).real.astype(np.float32)
        def gradient(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            gx = np.roll(img, -1, axis=1) - img
            gy = np.roll(img, -1, axis=0) - img
            return gx.astype(np.float32), gy.astype(np.float32)
        def divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
            tx = gx - np.roll(gx, 1, axis=1)
            ty = gy - np.roll(gy, 1, axis=0)
            return (tx + ty).astype(np.float32)
        for k in range(n_iterations):
            gx, gy = gradient(x)
            ux = soft_threshold(gx + eta_x / mu2, lambda_tv / mu2)
            uy = soft_threshold(gy + eta_y / mu2, lambda_tv / mu2)
            Mx = H_forward(x)
            v = (y + mu1 * Mx + xi) / (1.0 + mu1)
            if nonneg:
                w_var = np.maximum(x + rho / mu3, 0.0)
            else:
                w_var = x + rho / mu3
            rhs = (
                H_adjoint(mu1 * v - xi)
                + divergence(mu2 * ux - eta_x, mu2 * uy - eta_y)
                + (mu3 * w_var - rho)
            ).astype(np.float32)
            Xf = np.fft.rfft2(rhs) / (denom + self.cfg.eps)
            x = np.fft.irfft2(Xf, s=(h, w)).real.astype(np.float32)
            Mx_new = H_forward(x)
            xi = xi + mu1 * (Mx_new - v)
            gx_new, gy_new = gradient(x)
            eta_x = eta_x + mu2 * (gx_new - ux)
            eta_y = eta_y + mu2 * (gy_new - uy)
            rho = rho + mu3 * (x - w_var)
            if verbose and (k + 1) % 50 == 0:
                residual = np.linalg.norm(Mx_new - y)
                print(f"Iter {k+1}/{n_iterations}, residual: {residual:.4f}")
        return np.clip(x, 0.0, None).astype(np.float32)
    def reconstruct_rgb(
        self,
        measurement: np.ndarray,
        z_mm: float = 20.0,
        lambda_tv: float = 0.02,
        n_iterations: int = 200,
        verbose: bool = False
    ) -> np.ndarray:
        if measurement.ndim == 2:
            h, w = measurement.shape
            cfg_binned = DiffuserCamConfig(
                pixel_pitch_um=self.cfg.pixel_pitch_um * self.cfg.binning,
                diffuser_sensor_distance_mm=self.cfg.diffuser_sensor_distance_mm,
                diffuser_feature_um=self.cfg.diffuser_feature_um,
                aperture_hw_mm=self.cfg.aperture_hw_mm,
                seed=self.cfg.seed
            )
            cam_binned = DiffuserCam(cfg_binned)
            psf = cam_binned.generate_psf((h, w), z_mm)
            return self.reconstruct_2d_admm(measurement, psf, lambda_tv, n_iterations, verbose=verbose)
        h, w, c = measurement.shape
        result = np.zeros_like(measurement)
        cfg_binned = DiffuserCamConfig(
            pixel_pitch_um=self.cfg.pixel_pitch_um * self.cfg.binning,
            diffuser_sensor_distance_mm=self.cfg.diffuser_sensor_distance_mm,
            diffuser_feature_um=self.cfg.diffuser_feature_um,
            aperture_hw_mm=self.cfg.aperture_hw_mm,
            seed=self.cfg.seed
        )
        cam_binned = DiffuserCam(cfg_binned)
        wavelengths = [630.0, 530.0, 460.0]
        for i in range(c):
            if verbose:
                print(f"Reconstructing channel {['R', 'G', 'B'][i]}...")
            psf = cam_binned.generate_psf((h, w), z_mm, wavelength_nm=wavelengths[i])
            result[..., i] = self.reconstruct_2d_admm(
                measurement[..., i], psf, lambda_tv, n_iterations, verbose=verbose
            )
        return result
def process_image(
    input_path: str,
    output_dir: str,
    z_mm: float = 20.0,
    mode: str = "both",
    lambda_tv: float = 0.02,
    n_iterations: int = 100,
    verbose: bool = True
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(input_path)
    rgb = to_float01_rgb(img)
    basename = os.path.splitext(os.path.basename(input_path))[0]
    if verbose:
        print(f"Processing: {input_path}")
        print(f"  Size: {rgb.shape[0]}x{rgb.shape[1]}")
    cam = DiffuserCam()
    if mode in ("capture", "both"):
        measurement, debug = cam.capture(rgb, z_mm=z_mm, mode="rgb", do_binning=True)
        if "psf_g" in debug:
            psf_vis = normalize_01(debug["psf_g"])
            save_image(os.path.join(output_dir, f"{basename}_psf.png"), psf_vis)
        save_image(os.path.join(output_dir, f"{basename}_measurement.png"), measurement)
        if verbose:
            print(f"  Saved measurement: {basename}_measurement.png")
    if mode in ("reconstruct", "both"):
        if mode == "reconstruct":
            measurement, _ = cam.capture(rgb, z_mm=z_mm, mode="rgb", do_binning=True)
        if verbose:
            print(f"  Starting ADMM reconstruction (iterations: {n_iterations})...")
        reconstructed = cam.reconstruct_rgb(
            measurement, 
            z_mm=z_mm, 
            lambda_tv=lambda_tv,
            n_iterations=n_iterations,
            verbose=verbose
        )
        reconstructed = normalize_01(reconstructed)
        save_image(os.path.join(output_dir, f"{basename}_reconstructed.png"), reconstructed)
        if verbose:
            print(f"  Saved reconstruction: {basename}_reconstructed.png")
def batch_process(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.png",
    z_mm: float = 20.0,
    mode: str = "both",
    lambda_tv: float = 0.02,
    n_iterations: int = 100,
    verbose: bool = True
) -> None:
    image_paths = glob.glob(os.path.join(input_dir, pattern))
    image_paths += glob.glob(os.path.join(input_dir, pattern.replace("png", "jpg")))
    image_paths += glob.glob(os.path.join(input_dir, pattern.replace("png", "jpeg")))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        print(f"In {input_dir} no matches found for {pattern} ")
        return
    print(f"Found {len(image_paths)} images")
    print("=" * 60)
    for i, path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] ", end="")
        try:
            process_image(
                path, output_dir, z_mm, mode, lambda_tv, n_iterations, verbose
            )
        except Exception as e:
            print(f"Processing failed: {e}")
    print("\n" + "=" * 60)
    print(f"Processing completed! Results saved in: {output_dir}")
def parse_args():
    parser = argparse.ArgumentParser(
        description="DiffuserCam Software Implementation - Lensless 3D Imaging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python DiffuserCam.py --input image.png --output output_dir/ --mode both
  python DiffuserCam.py --input_dir ./images/ --output output_dir/ --mode capture
  python DiffuserCam.py --input image.png --output output_dir/ --mode capture

from __future__ import annotations
import argparse
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from PIL import Image
try:
    from dypp_v2 import DyPPCamera, DyPPCameraConfig, to_float01_rgb, save_image
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dypp_v2 import DyPPCamera, DyPPCameraConfig, to_float01_rgb, save_image
VIDEO_RE = re.compile(r"^S(?P<S>\d{3})C(?P<C>\d{3})P(?P<P>\d{3})R(?P<R>\d{3})A(?P<A>\d{3})$")
worker_camera: Optional[DyPPCamera] = None
def parse_video_name(video_dir_name: str) -> Optional[Dict[str, int]]:
    m = VIDEO_RE.match(video_dir_name)
    if not m:
        return None
    d = {k: int(v) for k, v in m.groupdict().items()}
    return d
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
    if protocol == "all": return True
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
        if split == "train": return meta["S"] in train_ss
        if split == "val": return meta["S"] in val_ss
        return (meta["S"] not in train_ss) and (meta["S"] not in val_ss)
    raise ValueError(f"Unknown protocol: {protocol}")
def list_frames(video_dir: str) -> List[str]:
    try:
        frames = [fn for fn in os.listdir(video_dir) if fn.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError:
        return []
    def _key(x: str) -> Tuple[int, str]:
        m = re.match(r"^(\d+)", x)
        return (int(m.group(1)) if m else 10**9, x)
    frames.sort(key=_key)
    return frames
def list_video_dirs(input_root: str) -> List[str]:
    if not os.path.exists(input_root): return []
    out = [name for name in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, name))]
    out.sort()
    return out
@dataclass(frozen=True)
class BuildConfig:
    input_root: str
    output_root: str
    protocol: str
    split: str
    split_file: str
    workers: int
    overwrite: bool
    alpha_opd_scale_m: float
    noise_sigma: float
    g_weights_npz: str
def worker_init(cfg: BuildConfig):
    global worker_camera
    cam_cfg = DyPPCameraConfig(
        dz=350,
        rgb_wavelengths_nm=(640.0, 550.0, 460.0),
        pixel_size_um=1.0,
        f_number=1.8,
        pupil_hw=(256, 256),
        noise_sigma=float(cfg.noise_sigma),
        alpha_opd_scale_m=float(cfg.alpha_opd_scale_m),
    )
    worker_camera = DyPPCamera(cam_cfg=cam_cfg, mlp_seed=42)
    if cfg.g_weights_npz:
        worker_camera.mlp.load_npz(cfg.g_weights_npz)
def process_one_video(video_name: str, cfg: BuildConfig, splits: Dict[str, List[int]]) -> Tuple[str, int, int, bool]:
    global worker_camera
    if worker_camera is None:
        worker_init(cfg)
    meta = parse_video_name(video_name)
    if meta is None: return (video_name, -1, 0, False)
    if not is_in_split(meta, splits, cfg.protocol, cfg.split): return (video_name, meta["A"] - 1, 0, False)
    in_dir = os.path.join(cfg.input_root, video_name)
    out_dir = os.path.join(cfg.output_root, video_name)
    label = meta["A"] - 1
    if (not cfg.overwrite) and os.path.isdir(out_dir):
        try:
            with os.scandir(out_dir) as it:
                if any(entry.name.lower().endswith(('.jpg', '.png')) for entry in it):
                    return (video_name, label, 0, True)
        except Exception:
            pass
    frames = list_frames(in_dir)
    if not frames: return (video_name, label, 0, False)
    os.makedirs(out_dir, exist_ok=True)
    wrote = 0
    for fn in frames:
        out_path = os.path.join(out_dir, fn)
        if (not cfg.overwrite) and os.path.exists(out_path):
            wrote += 1
            continue
        try:
            in_path = os.path.join(in_dir, fn)
            img = Image.open(in_path)
            rgb = to_float01_rgb(img)
            seed = stable_int_seed(video_name, fn)
            out = worker_camera.capture(rgb, seed=seed)
            save_image(out_path, out)
            wrote += 1
        except Exception:
            continue
    return (video_name, label, wrote, False)
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", type=str, default="/data2/NTU_data/resize_all_gt")
    p.add_argument("--output_root", type=str, default="/data2/NTU_data/resize_all_gt_dypp_2")
    p.add_argument("--protocol", type=str, default="all")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--split_file", type=str, default=os.path.join(os.path.dirname(__file__), "NTURGBD_split.txt"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--alpha_opd_scale_m", type=float, default=1e-6)
    p.add_argument("--noise_sigma", type=float, default=0.0)
    p.add_argument("--g_weights_npz", type=str, default="")
    p.add_argument("--write_manifest", action="store_true")
    args = p.parse_args()
    cfg = BuildConfig(
        input_root=args.input_root, output_root=args.output_root, protocol=args.protocol,
        split=args.split, split_file=args.split_file, workers=int(args.workers),
        overwrite=bool(args.overwrite), alpha_opd_scale_m=float(args.alpha_opd_scale_m),
        noise_sigma=float(args.noise_sigma), g_weights_npz=str(args.g_weights_npz),
    )
    splits = load_ntu_splits(cfg.split_file)
    videos = list_video_dirs(cfg.input_root)
    if args.reverse:
        print(f"[{os.getpid()}] REVERSE mode.")
        videos.reverse()
    else:
        print(f"[{os.getpid()}] FORWARD mode.")
    print(f"Found {len(videos)} videos. Workers: {cfg.workers}")
    os.makedirs(cfg.output_root, exist_ok=True)
    processed = 0
    kept = 0
    skipped_count = 0
    with ProcessPoolExecutor(max_workers=cfg.workers, initializer=worker_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(process_one_video, v, cfg, splits) for v in videos]
        for fut in as_completed(futs):
            _, _, wrote, skipped = fut.result()
            processed += 1
            if skipped: skipped_count += 1
            if wrote > 0: kept += 1
            if processed % 100 == 0:
                print(f"[{'REV' if args.reverse else 'FWD'}][{processed}/{len(videos)}] kept={kept}, skipped={skipped_count}")
    print("Done.")
if __name__ == "__main__":
    main()
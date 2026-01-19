import numpy as np
import cv2
import os
from numpy.fft import fft2, ifft2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import time
class CodedApertureCamera:
    def __init__(self, mask_transmittance=0.5):
        self.transmittance = mask_transmittance
        self._cached_mask = None
        self._cached_resolution = None
    def _generate_pseudorandom_mask(self, height, width):
        random_matrix = np.random.rand(height, width)
        binary_mask = (random_matrix < self.transmittance).astype(np.float32)
        return binary_mask
    def _get_mask(self, height, width):
        if self._cached_mask is None or self._cached_resolution != (height, width):
            self._cached_mask = self._generate_pseudorandom_mask(height, width)
            self._cached_resolution = (height, width)
        return self._cached_mask
    def preprocess_image(self, image):
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        return image_gray.astype(np.float32) / 255.0
    def capture(self, image, add_noise=False):
        obj = self.preprocess_image(image)
        height, width = obj.shape
        psf = self._get_mask(height, width)
        O_f = fft2(obj)
        A_f = fft2(psf)
        D_f = O_f * A_f
        captured_image = np.real(ifft2(D_f))
        if add_noise:
            noise = np.random.normal(0, 0.01, captured_image.shape)
            captured_image += noise
        captured_image = (captured_image - captured_image.min()) / (captured_image.max() - captured_image.min() + 1e-8)
        return captured_image, obj, psf
def process_single_frame(args):
    input_path, output_path, ca_camera = args
    try:
        image = cv2.imread(input_path)
        if image is None:
            return (False, input_path, "Cannot read image")
        ca_result, _, _ = ca_camera.capture(image)
        ca_result_uint8 = (ca_result * 255).astype(np.uint8)
        cv2.imwrite(output_path, ca_result_uint8)
        return (True, input_path, None)
    except Exception as e:
        return (False, input_path, str(e))
def process_video_folder(video_folder, input_base, output_base, ca_camera, image_extensions):
    input_video_dir = os.path.join(input_base, video_folder)
    output_video_dir = os.path.join(output_base, video_folder)
    os.makedirs(output_video_dir, exist_ok=True)
    try:
        image_files = [f for f in os.listdir(input_video_dir) 
                       if os.path.splitext(f)[1].lower() in image_extensions]
    except Exception as e:
        return (video_folder, 0, 0, str(e))
    success_count = 0
    fail_count = 0
    for filename in image_files:
        input_path = os.path.join(input_video_dir, filename)
        output_path = os.path.join(output_video_dir, filename)
        try:
            image = cv2.imread(input_path)
            if image is None:
                fail_count += 1
                continue
            ca_result, _, _ = ca_camera.capture(image)
            ca_result_uint8 = (ca_result * 255).astype(np.uint8)
            cv2.imwrite(output_path, ca_result_uint8)
            success_count += 1
        except Exception as e:
            fail_count += 1
    return (video_folder, success_count, fail_count, None)
def process_ntu_dataset(input_dir, output_dir, mask_transmittance=0.5, num_workers=16, seed=42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    video_folders = [f for f in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, f))]
    video_folders = sorted(video_folders)
    print(f"=" * 60)
    print(f"NTU RGB+D 120 Dataset Processing (CVPRW19 Coded Aperture)")
    print(f"=" * 60)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Video folders: {len(video_folders)}")
    print(f"Workers: {num_workers}")
    print(f"Mask transmittance: {mask_transmittance}")
    print(f"Random seed: {seed}")
    print(f"=" * 60)
    total_success = 0
    total_fail = 0
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for video_folder in video_folders:
            ca_camera = CodedApertureCamera(mask_transmittance=mask_transmittance)
            future = executor.submit(
                process_video_folder,
                video_folder, input_dir, output_dir, ca_camera, image_extensions
            )
            futures[future] = video_folder
        with tqdm(total=len(video_folders), desc="Processing videos", unit="video") as pbar:
            for future in as_completed(futures):
                video_folder, success, fail, error = future.result()
                total_success += success
                total_fail += fail
                if error:
                    tqdm.write(f"[Error] {video_folder}: {error}")
                pbar.update(1)
                pbar.set_postfix({
                    "Success": total_success,
                    "Failed": total_fail
                })
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"Processed!")
    print(f"=" * 60)
    print(f"Total videos: {len(video_folders)}")
    print(f"Success frames: {total_success}")
    print(f"Failed frames: {total_fail}")
    print(f"Total time: {elapsed_time:.2f} sec")
    print(f"Avg speed: {total_success / elapsed_time:.2f} frames/sec")
    print(f"Output dir: {output_dir}")
    print(f"=" * 60)
def process_directory(input_dir, output_dir, mask_transmittance=0.5):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    ca_camera = CodedApertureCamera(mask_transmittance=mask_transmittance)
    image_files = [f for f in os.listdir(input_dir) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    print(f"Found {len(image_files)} images to process")
    for idx, filename in enumerate(sorted(image_files)):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"[{idx+1}/{len(image_files)}] Skipped: {filename} (Cannot read)")
            continue
        original_shape = image.shape[:2]
        ca_result, _, _ = ca_camera.capture(image)
        ca_result_uint8 = (ca_result * 255).astype(np.uint8)
        cv2.imwrite(output_path, ca_result_uint8)
        print(f"[{idx+1}/{len(image_files)}] Processed: {filename} (resolution: {original_shape[1]}x{original_shape[0]})")
    print(f"\nAll processed! Output dir: {output_dir}")
if __name__ == "__main__":
    input_dir = "/data2/NTU_data/resize_all_gt"
    output_dir = "/data2/NTU_data/resize_all_gt_cvprw19"
    process_ntu_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        mask_transmittance=0.5,
        num_workers=32,
        seed=42
    )

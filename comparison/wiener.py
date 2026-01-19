import os
import numpy as np
from PIL import Image
from skimage import img_as_float, img_as_ubyte
from skimage.restoration import wiener
import time
input_dir = '/data2/NTU_data/resize_all_mb'
output_dir = '/data2/NTU_data/resize_all_mb_wiener'
PSF_SIZE = 21
PSF_SIGMA = 4.0
BALANCE = 0.19
MAX_IMAGES = 5100
def create_gaussian_psf(size, sigma):
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, y)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf
PSF = create_gaussian_psf(PSF_SIZE, PSF_SIGMA)
def apply_wiener_filter(image_float):
    if len(image_float.shape) == 2:
        return wiener(image_float, PSF, balance=BALANCE)
    elif len(image_float.shape) == 3:
        result = np.zeros_like(image_float)
        for c in range(image_float.shape[2]):
            result[:, :, c] = wiener(image_float[:, :, c], PSF, balance=BALANCE)
        return result
    else:
        raise ValueError(f"Unsupported image shape: {image_float.shape}")
def safe_convert_to_ubyte(image):
    image_clipped = np.clip(image, 0, 1)
    return img_as_ubyte(image_clipped)
def process_single_image(input_path, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        img_float = img_as_float(img_array)
        filtered = apply_wiener_filter(img_float)
        filtered_ubyte = safe_convert_to_ubyte(filtered)
        Image.fromarray(filtered_ubyte).save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error: {input_path} - {e}")
        return False
def main():
    print("=" * 60)
    print("Batch Wiener Filter (Single Thread)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Wiener params: PSF={PSF_SIZE}, sigma={PSF_SIGMA}, balance={BALANCE}")
    print(f"Max images: {MAX_IMAGES if MAX_IMAGES else 'unlimited'}")
    print("=" * 60)
    start_time = time.time()
    total_images = 0
    total_folders = 0
    errors = 0
    folders = sorted(os.listdir(input_dir))
    num_folders = len(folders)
    print(f"Total {num_folders} folders, processing...\n")
    stopped_early = False
    for folder_idx, folder_name in enumerate(folders):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        filenames = sorted([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        folder_count = 0
        for filename in filenames:
            if MAX_IMAGES is not None and total_images >= MAX_IMAGES:
                stopped_early = True
                break
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_dir, folder_name, filename)
            if process_single_image(input_path, output_path):
                folder_count += 1
                total_images += 1
            else:
                errors += 1
        total_folders += 1
        elapsed = time.time() - start_time
        speed = total_images / elapsed if elapsed > 0 else 0
        print(f"[{folder_idx+1}/{num_folders}] {folder_name} ({folder_count} imgs) "
              f"| Total: {total_images} imgs | {speed:.1f} imgs/sec")
        if stopped_early:
            print(f"\nReached max images limit ({MAX_IMAGES}), stopping")
            break
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print("Wiener filter processing completed!")
    print(f"Folders: {total_folders}")
    print(f"Success: {total_images}")
    print(f"Failed: {errors}")
    print(f"Total time: {elapsed_total/60:.1f} min")
    print(f"Avg speed: {total_images/elapsed_total:.1f} imgs/sec")
if __name__ == "__main__":
    main()

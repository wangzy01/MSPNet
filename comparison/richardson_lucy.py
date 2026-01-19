import os
import numpy as np
from PIL import Image
from skimage import img_as_float, img_as_ubyte
from skimage.restoration import richardson_lucy
import time
input_dir = '/data2/NTU_data/resize_all_mb'
output_dir = '/data2/NTU_data/resize_all_mb_richardson_lucy'
MAX_IMAGES = 5100
ITERATIONS_LIST = [5, 10, 15, 30]
PSF_CONFIGS = [
    (11, 2.5),
    (15, 3.5),
    (21, 5.0),
    (25, 6.0),
]
def create_gaussian_psf(size, sigma):
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, y)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf
def apply_richardson_lucy(image_float, psf, num_iter):
    if len(image_float.shape) == 2:
        return richardson_lucy(image_float, psf, num_iter=num_iter)
    elif len(image_float.shape) == 3:
        result = np.zeros_like(image_float)
        for c in range(image_float.shape[2]):
            result[:, :, c] = richardson_lucy(image_float[:, :, c], psf, num_iter=num_iter)
        return result
    else:
        raise ValueError(f"Unsupported image shape: {image_float.shape}")
def safe_convert_to_ubyte(image):
    image_clipped = np.clip(image, 0, 1)
    return img_as_ubyte(image_clipped)
def process_single_image(input_path, output_path, psf, num_iter):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        img_float = img_as_float(img_array)
        deconvolved = apply_richardson_lucy(img_float, psf, num_iter)
        deconvolved_ubyte = safe_convert_to_ubyte(deconvolved)
        Image.fromarray(deconvolved_ubyte).save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"  Error: {input_path} - {e}")
        return False
def main():
    print("=" * 70)
    print("Batch Richardson-Lucy Deblur (Single Thread)")
    print("=" * 70)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Iterations: {ITERATIONS_LIST}")
    print(f"PSF config: {PSF_CONFIGS}")
    print(f"Max images: {MAX_IMAGES if MAX_IMAGES else 'unlimited'}")
    print("=" * 70)
    start_time = time.time()
    total_images = 0
    total_folders = 0
    errors = 0
    psf_size, psf_sigma = PSF_CONFIGS[0]
    num_iter = ITERATIONS_LIST[0]
    psf = create_gaussian_psf(psf_size, psf_sigma)
    print(f"Using PSF({psf_size}, σ={psf_sigma}), iter={num_iter}")
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
            if process_single_image(input_path, output_path, psf, num_iter):
                folder_count += 1
                total_images += 1
            else:
                errors += 1
        total_folders += 1
        elapsed = time.time() - start_time
        speed = total_images / elapsed if elapsed > 0 else 0
        print(f"[{folder_idx+1}/{num_folders}] {folder_name} ({folder_count}imgs) "
              f"| Total: {total_images}imgs | {speed:.1f}imgs/sec")
        if stopped_early:
            print(f"\nReached max images ({MAX_IMAGES} imgs), stopping")
            break
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print("Richardson-Lucy deblur completed!")
    print(f"Folders: {total_folders} ")
    print(f"Success: {total_images} imgs")
    print(f"Failed: {errors} imgs")
    print(f"Total time: {elapsed_total/60:.1f} min")
    print(f"Avg speed: {total_images/elapsed_total:.1f} imgs/sec")
if __name__ == "__main__":
    main()
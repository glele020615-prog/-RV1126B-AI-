import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
FIXED_VARIANTS = ['original', 'low_light', 'blur', 'noise', 'compress', 'enhanced']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a representative RKNN quantization dataset with original, degraded, and enhanced images.'
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        required=True,
        help='One or more source folders containing candidate images.'
    )
    parser.add_argument(
        '--output-dir',
        default='quantization_dataset',
        help='Directory to write generated images.'
    )
    parser.add_argument(
        '--dataset-file',
        default='dataset.txt',
        help='Path of the output dataset list file.'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=300,
        help='Number of source images to sample before augmentation.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling.'
    )
    parser.add_argument(
        '--sizes',
        nargs='+',
        type=int,
        default=[1088],
        help='Resize target sizes used for the exported images. The default is 1088.'
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        default=FIXED_VARIANTS,
        help='Image variants to generate. The default is a fixed balanced recipe.'
    )
    parser.add_argument(
        '--include-original',
        action='store_true',
        help='Also copy the original sampled image into the output set.'
    )
    return parser.parse_args()


def collect_images(source_directories):
    collected_paths = []
    for source_directory in source_directories:
        source_path = Path(source_directory)
        if not source_path.exists():
            continue
        for current_root, _, filenames in os.walk(source_path):
            for filename in filenames:
                file_path = Path(current_root) / filename
                if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    collected_paths.append(file_path)
    return collected_paths


def ensure_dir(directory_path):
    directory_path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(file_path):
    parent_directory = file_path.parent
    if parent_directory and parent_directory != Path('.'):
        ensure_dir(parent_directory)


def read_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Failed to read image: {image_path}')
    return image


def resize_to_square(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def apply_gamma(image, gamma_value):
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma_value)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def apply_low_light(image, rng):
    gamma_value = rng.uniform(1.6, 2.4)
    darker = apply_gamma(image, gamma_value)
    alpha = rng.uniform(0.45, 0.75)
    beta = rng.uniform(-18, 8)
    adjusted = cv2.convertScaleAbs(darker, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.75, 0.95)
    hsv[:, :, 2] *= rng.uniform(0.6, 0.85)
    hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_blur(image, rng):
    if rng.random() < 0.5:
        kernel_size = rng.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=rng.uniform(0.8, 2.2))
    kernel_size = rng.choice([3, 5])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    angle = rng.choice([0, 45, 90, 135])
    rotation_matrix = cv2.getRotationMatrix2D((kernel_size / 2.0, kernel_size / 2.0), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def add_gaussian_noise(image, rng):
    noise_std = rng.uniform(8.0, 24.0)
    noise = rng.normal(0.0, noise_std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_compression(image, rng):
    quality = int(rng.uniform(18, 60))
    success, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_enhancement(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def build_variant(image, variant_name, rng):
    if variant_name == 'original':
        return image
    if variant_name == 'low_light':
        return apply_low_light(image, rng)
    if variant_name == 'blur':
        return apply_blur(image, rng)
    if variant_name == 'noise':
        return add_gaussian_noise(image, rng)
    if variant_name == 'compress':
        return apply_compression(image, rng)
    if variant_name == 'enhanced':
        return apply_enhancement(image)
    raise ValueError(f'Unsupported variant: {variant_name}')


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    source_images = collect_images(args.sources)
    if not source_images:
        raise SystemExit('No images found in the provided source folders.')

    rng.shuffle(source_images)
    selected_images = source_images[: min(args.count, len(source_images))]

    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    dataset_paths = []
    variants = list(args.variants)
    target_sizes = sorted(set(args.sizes))
    variant_counts = {variant_name: 0 for variant_name in variants}

    for index, image_path in enumerate(selected_images):
        image = read_image(image_path)
        for size in target_sizes:
            resized_image = resize_to_square(image, size)
            base_name = f'{index:05d}_{image_path.stem}_{size}'

            if args.include_original:
                original_path = output_root / 'original' / f'{base_name}_original.jpg'
                ensure_dir(original_path.parent)
                cv2.imwrite(str(original_path), resized_image)
                dataset_paths.append(str(original_path.resolve()))
                variant_counts['original'] = variant_counts.get('original', 0) + 1

            for variant_name in variants:
                if variant_name == 'original' and args.include_original:
                    continue
                variant_image = build_variant(resized_image.copy(), variant_name, rng)
                variant_path = output_root / variant_name / f'{base_name}_{variant_name}.jpg'
                ensure_dir(variant_path.parent)
                cv2.imwrite(str(variant_path), variant_image)
                dataset_paths.append(str(variant_path.resolve()))
                variant_counts[variant_name] = variant_counts.get(variant_name, 0) + 1

    dataset_file_path = Path(args.dataset_file)
    ensure_parent_dir(dataset_file_path)
    with open(dataset_file_path, 'w', encoding='utf-8') as dataset_file:
        for image_path in dataset_paths:
            dataset_file.write(image_path + '\n')

    print(f'Collected source images: {len(source_images)}')
    print(f'Selected images: {len(selected_images)}')
    print(f'Generated calibration images: {len(dataset_paths)}')
    print('Variant counts:')
    for variant_name in variants:
        print(f'  - {variant_name}: {variant_counts.get(variant_name, 0)}')
    print(f'Dataset list saved to: {dataset_file_path.resolve()}')


if __name__ == '__main__':
    main()
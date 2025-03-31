import os
import cv2
import random
import numpy as np

# Set this flag to True if you want to convert augmented images to greyscale.
CONVERT_TO_GREYSCALE = True
# Set this flag if you want to boost contrast during conversion
INCREASE_CONTRAST = True

# ---------------------------
# Helper Function: Convert to Greyscale
# ---------------------------
def convert_to_greyscale(img, increase_contrast=False):
    """Convert a BGR image to greyscale, and optionally increase contrast using histogram equalization."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if increase_contrast:
        gray = cv2.equalizeHist(gray)
    return gray

# ---------------------------
# Augmentation Functions
# ---------------------------
def augment_translation(img):
    h, w = img.shape[:2]
    max_tx = int(0.1 * w)
    max_ty = int(0.1 * h)
    tx = random.randint(-max_tx, max_tx)
    ty = random.randint(-max_ty, max_ty)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))

def augment_rotation(img):
    h, w = img.shape[:2]
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def augment_stretching(img):
    h, w = img.shape[:2]
    sx = random.uniform(0.8, 1.2)
    M = np.float32([[sx, 0, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))

def augment_shearing(img):
    h, w = img.shape[:2]
    shear_factor = random.uniform(-0.2, 0.2)
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))

def augment_lens_distortion(img):
    h, w = img.shape[:2]
    K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
    distCoeffs = np.array([0.1, -0.05, 0, 0], dtype=np.float32)
    return cv2.undistort(img, K, distCoeffs)

def augment_cutout(img):
    h, w = img.shape[:2]
    cutout_w = int(0.2 * w)
    cutout_h = int(0.2 * h)
    x = random.randint(0, w - cutout_w)
    y = random.randint(0, h - cutout_h)
    img_copy = img.copy()
    img_copy[y:y+cutout_h, x:x+cutout_w] = 0
    return img_copy

def augment_color_jitter(img):
    img_copy = img.astype(np.float32)
    brightness = random.uniform(-50, 50)
    contrast = random.uniform(0.8, 1.2)
    img_jitter = img_copy * contrast + brightness
    img_jitter = np.clip(img_jitter, 0, 255).astype(np.uint8)
    return img_jitter

def get_augmented_images(img):
    """Return a dictionary mapping augmentation names to the augmented image."""
    augmentations = {
        "translation": augment_translation,
        "rotation": augment_rotation,
        "stretching": augment_stretching,
        "shearing": augment_shearing,
        "lens_distortion": augment_lens_distortion,
        "cutout": augment_cutout,
        "color_jitter": augment_color_jitter,
    }
    aug_images = {}
    for aug_name, func in augmentations.items():
        try:
            aug_images[aug_name] = func(img)
        except Exception as e:
            print(f"Augmentation {aug_name} failed: {e}")
    return aug_images


# ---------------------------
# Original Greyscale Saving Function
# ---------------------------
def save_original_greyscale():
    """
    Convert each resized image (without augmentation) to greyscale and save it in a dedicated folder.
    """
    input_folder = "dl_project/preprocess/RealESRGAN_x4plus_train_imgs"
    output_folder = "dl_project/preprocess/augmented_RealESRGAN_x4plus_train_GREYSCALE/pure"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {filename} (unable to load image).")
                continue
            gray_img = convert_to_greyscale(img, increase_contrast=INCREASE_CONTRAST)
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, gray_img)
            print(f"Saved original greyscale image: {out_path}")
    print("Finished saving original greyscale images.")

# ---------------------------
# Data Augmentation Function
# ---------------------------
def apply_augmentations():
    # Input folder: resized images produced by resize_crops.py
    input_folder = "dl_project/preprocess/RealESRGAN_x4plus_train_imgs"
    # Base folder for augmented images
    base_output_folder = "dl_project/preprocess/augmented_RealESRGAN_x4plus_train_GREYSCALE"  #TODO: file name should come from the input path
    os.makedirs(base_output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {filename} (unable to load image).")
                continue

            # Apply augmentations
            augmented_dict = get_augmented_images(img)
            for aug_name, aug_img in augmented_dict.items():
                # Optionally convert the augmented image to greyscale
                if CONVERT_TO_GREYSCALE:
                    aug_img = convert_to_greyscale(aug_img, increase_contrast=INCREASE_CONTRAST)
                # Create a dedicated folder for this augmentation method
                output_dir = os.path.join(base_output_folder, aug_name)
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, aug_img)
                print(f"Saved {aug_name} augmented image: {out_path}")
    print("Finished data augmentation.")

if __name__ == "__main__":
    # if CONVERT_TO_GREYSCALE:
    #     save_original_greyscale()
    apply_augmentations()

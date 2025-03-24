import os
import shutil


def group_images_by_track():
    # Define source and destination folders.
    src_folder = "dl_project/prepossess/RealESRGAN_x4plus_test_imgs"
    dst_folder = "dl_project/reorganize/RealESRGAN_x4plus_test_imgs_by_track"

    # Create the destination folder if it doesn't exist.
    os.makedirs(dst_folder, exist_ok=True)

    # Iterate over each file in the source folder.
    for filename in os.listdir(src_folder):
        if filename.lower().endswith(".jpg"):
            # Expected format: {track number}_{id}_out.jpg
            parts = filename.split("_")
            if len(parts) < 3:
                print(f"Skipping file with unexpected naming format: {filename}")
                continue
            track_number = parts[0]  # Extract track number from the filename.

            # Create a subfolder for this track number.
            track_folder = os.path.join(dst_folder, track_number)
            os.makedirs(track_folder, exist_ok=True)

            # Define full paths for source and destination.
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(track_folder, filename)

            # Copy the image from the source to the track subfolder.
            shutil.copy(src_path, dst_path)
            print(f"Copied {filename} to {track_folder}")


if __name__ == "__main__":
    group_images_by_track()

import os
import cv2

def resize_images():
    # Folder paths
    input_folder = "out/SoccerNetResults/crops/imgs"
    output_folder = "dl_project/prepossess/resized_imgs"  # TODO: file name should come from the input path
    os.makedirs(output_folder, exist_ok=True)

    # Read stats from the stats file
    stats_file = "dl_project/prepossess/crops_image_stats.txt"
    try:
        with open(stats_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Stats file '{stats_file}' not found. Please run analyze_crops.py first.")
        return

    target_width = None
    target_height = None
    for line in lines:
        if "Overall Median Width:" in line:
            target_width = int(line.split(":")[1].strip())
        elif "Overall Median Height:" in line:
            target_height = int(line.split(":")[1].strip())

    if target_width is None or target_height is None:
        print("Could not parse target dimensions from stats file.")
        return

    print(f"Resizing images to: {target_width} x {target_height}")

    # Process each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {filename} (unable to load image).")
                continue
            resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, resized_img)
            print(f"Saved resized image: {out_path}")

if __name__ == "__main__":
    resize_images()

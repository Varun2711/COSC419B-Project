import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import statistics  # Added to compute median

# Define the folder containing the images
folder_path = "dl_project/train_crops/imgs"

# Output file name
output_file_name = "train_crops_image_stats.txt" # TODO: file name should come from the input path

# Dictionary to store images grouped by track number.
# Each key is a track number and its value is a list of tuples: (filename, width, height)
track_groups = defaultdict(list)
# Lists to collect dimensions across all images
all_widths = []
all_heights = []

def collect_crops_info():
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            # Expected filename format: {track number}_{id}.jpg
            parts = filename.split("_")
            if len(parts) < 2:
                print(f"Skipping file with unexpected naming format: {filename}")
                continue

            track_number = parts[0]
            image_path = os.path.join(folder_path, filename)

            # Open image to get its resolution
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading image {filename}: {e}")
                continue

            # Append the result to the corresponding track group
            track_groups[track_number].append((filename, width, height))
            all_widths.append(width)
            all_heights.append(height)

def compute_key_info_and_save():
    # Compute overall statistics across all images
    if all_widths and all_heights:
        overall_max_width = max(all_widths)
        overall_max_height = max(all_heights)
        overall_median_width = statistics.median(all_widths)
        overall_median_height = statistics.median(all_heights)
        overall_mean_width = statistics.mean(all_widths)
        overall_mean_height = statistics.mean(all_heights)
    else:
        overall_max_width = overall_max_height = overall_median_width = overall_median_height = overall_mean_width = overall_mean_height = None

    # Display the grouped results
    # for track, images in track_groups.items():
    #     # Extract widths and heights into separate lists
    #     widths = [info[1] for info in images]
    #     heights = [info[2] for info in images]
    #
    #     # Compute min, max and mean for widths
    #     min_width = min(widths)
    #     max_width = max(widths)
    #     mean_width = sum(widths) / len(widths)
    #     median_width = statistics.median(widths)
    #
    #     # Compute min, max and mean for heights
    #     min_height = min(heights)
    #     max_height = max(heights)
    #     mean_height = sum(heights) / len(heights)
    #     median_height = statistics.median(heights)
    #
    #     overall_max_width = max(overall_max_width, max_width)
    #     overall_max_height = max(overall_max_height, max_height)
    #
    #     # Display the computed statistics
    #     print(f"Track: {track}")
    #     print(f"  Width - min: {min_width}, max: {max_width}, mean: {mean_width:.2f}")
    #     print(f"  Height - min: {min_height}, max: {max_height}, mean: {mean_height:.2f}")
    #     print()

    # Display overall statistics
    print(f"Overall Max Width: {overall_max_width}")
    print(f"Overall Max Height: {overall_max_height}")
    print(f"Overall Median Width: {overall_median_width}")
    print(f"Overall Median Height: {overall_median_height}")
    print(f"Overall Mean Width: {overall_mean_width}")
    print(f"Overall Mean Height: {overall_mean_height}")
    print()

    # Save overall statistics to a file in dl_project/preprocess/
    output_dir = "dl_project/preprocess"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file_name)
    with open(output_file, "w") as f:
        f.write(f"Overall Max Width: {overall_max_width}\n")
        f.write(f"Overall Max Height: {overall_max_height}\n")
        f.write(f"Overall Median Width: {overall_median_width}\n")
        f.write(f"Overall Median Height: {overall_median_height}\n")
        f.write(f"Overall Mean Width: {overall_mean_width}\n")
        f.write(f"Overall Mean Height: {overall_mean_height}\n")

def visualize_and_save():
    # Draw a scatter plot of each image's width vs. height.
    plt.figure(figsize=(10, 6))
    for track, images in track_groups.items():
        widths = [info[1] for info in images]
        heights = [info[2] for info in images]
        plt.scatter(widths, heights, color='red', alpha=0.1)

    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("Distribution of Image Widths and Heights")
    plt.grid(True)
    plt.savefig("train crops distribution") # TODO: file name should come from the input path
    # plt.show()

if __name__ == '__main__':
    collect_crops_info()
    compute_key_info_and_save()
    visualize_and_save()
import json
import os

json_path = "dl_project/test_sr_LC_0.01/legible.json"

# Load the JSON data from the file
with open(json_path, 'r') as f:
    data = json.load(f)

# List to store just the file names
file_names = []

# Iterate over each key (each representing a group)
for key in data:
    for file_path in data[key]:
        # Extract only the file name
        file_names.append(os.path.basename(file_path))

# Save the file names list into a new JSON file
with open('dl_project/test_sr_LC_0.01/legible_sr.json', 'w') as outfile:
    json.dump(file_names, outfile, indent=4)

print("File names have been saved to file_names.json")

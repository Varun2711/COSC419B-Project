import json

gt_path = "../data/SoccerNet/test/test_gt.json"

output_json = "out/output_no_2pass.json"

# laod output
with open(output_json, "r") as f:
	output = json.load(f)

# load gt
with open(gt_path, "r") as f:
	gt = json.load(f)

correct = 0
for key in gt.keys():
	if gt[key] == int(output.get(key, -1)): # if key not in output, return -1
		correct += 1


total_acc = correct / len(gt)

print(f"Threshold: no 2-pass, accuracy: {total_acc * 100:.2f}%")
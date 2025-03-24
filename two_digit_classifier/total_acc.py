import json

gt_path = "../data/SoccerNet/test/test_gt.json"

output_json = "out/output_0.005.json"

# lc_out = "../out/SoccerNetResults/legible.json"
lc_out = "out/legible_sr_0.005.json"

# laod output
with open(output_json, "r") as f:
	output = json.load(f)

# load gt
with open(gt_path, "r") as f:
	gt = json.load(f)

with open(lc_out, "r") as f:
	lc = json.load(f)
	if type(lc) == type([]):
		lc_dict = {}
		for item in lc:
			group_id = item.split("_")[0]
			lc_dict[group_id] = 1
		lc = lc_dict

correct = 0
for key in gt.keys():
	if gt[key] == int(output.get(key, -1)): # if key not in output, return -1
		correct += 1
		if gt[key] == -1 and lc.get(key, "not found") != "not found":
			correct -= 1



total_acc = correct / len(gt)

print(f"Threshold: 0.005, accuracy: {total_acc * 100:.2f}%")
import argparse
import os
import legibility_classifier as lc
import numpy as np
import json
import helpers
from tqdm import tqdm
import configuration as config
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

def get_soccer_net_legibility_results(args, use_filtered=False, filter='sim', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    
    # Handle single tracklet
    if args.tracklet:
        tracklets = [args.tracklet]
    else:
        tracklets = os.listdir(path_to_images)

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(
                config.dataset['SoccerNet']['working_dir'],
                config.dataset['SoccerNet'][args.part]['sim_filtered']
            )
        else:
            path_to_filter_results = os.path.join(
                config.dataset['SoccerNet']['working_dir'],
                config.dataset['SoccerNet'][args.part]['gauss_filtered']
            )
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    if exclude_balls and not args.tracklet:  # Skip ball exclusion if processing single tracklet
        updated_tracklets = []
        soccer_ball_list = os.path.join(
            config.dataset['SoccerNet']['working_dir'],
            config.dataset['SoccerNet'][args.part]['soccer_ball_list']
        )
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if track not in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    def process_tracklet(directory):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(
            images_full_path,
            config.dataset['SoccerNet']['legibility_model'],
            arch=config.dataset['SoccerNet']['legibility_model_arch'],
            threshold=0.5
        )
        legible = list(np.nonzero(track_results)[0])
        if not legible:
            return (directory, None)
        else:
            legible_images = [images_full_path[i] for i in legible]
            return (directory, legible_images)

    legible_tracklets = {}
    illegible_tracklets = []
    
    print("Processing tracklets...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_tracklet, tracklets), total=len(tracklets)))
    
    for directory, legible_images in results:
        if legible_images is None:
            illegible_tracklets.append(directory)
        else:
            legible_tracklets[directory] = legible_images

    # Save results
    full_legible_path = os.path.join(
        config.dataset['SoccerNet']['working_dir'],
        config.dataset['SoccerNet'][args.part]['legible_result']
    )
    with open(full_legible_path, "w") as outfile:
        json.dump(legible_tracklets, outfile, indent=4)

    full_illegible_path = os.path.join(
        config.dataset['SoccerNet']['working_dir'],
        config.dataset['SoccerNet'][args.part]['illegible_result']
    )
    with open(full_illegible_path, "w") as outfile:
        json.dump({'illegible': illegible_tracklets}, outfile, indent=4)

    return legible_tracklets, illegible_tracklets

def generate_json_for_pose_estimator(args, legible=None):
    all_files = []
    if legible is not None:
        for key in legible.keys():
            for entry in legible[key]:
                all_files.append(os.path.join(os.getcwd(), entry))
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset['SoccerNet']['root_dir'])
        image_dir = config.dataset['SoccerNet'][args.part]['images']
        path_to_images = os.path.join(root_dir, image_dir)
        tracks = [args.tracklet] if args.tracklet else os.listdir(path_to_images)
        for tr in tracks:
            track_dir = os.path.join(path_to_images, tr)
            imgs = os.listdir(track_dir)
            for img in imgs:
                all_files.append(os.path.join(track_dir, img))

    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['pose_input_json'])
    helpers.generate_json(all_files, output_json)

def soccer_net_pipeline(args):
    Path(config.dataset['SoccerNet']['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True
    image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['images'])
    features_dir = config.dataset['SoccerNet'][args.part]['feature_output_folder']

    # Skip soccer ball filter for single tracklet
    if args.tracklet:
        args.pipeline['soccer_ball_filter'] = False

    if args.pipeline['soccer_ball_filter']:
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        if not os.path.exists(soccer_ball_list):
            print("Generating soccer ball list...")
            success = helpers.identify_soccer_balls(image_dir, soccer_ball_list)
        print("Done soccer ball filtering")

    if args.pipeline['feat']:
        print("Generating features...")
        if args.tracklet:
            with tempfile.TemporaryDirectory() as temp_dir:
                tracklet_path = os.path.join(image_dir, args.tracklet)
                if os.path.exists(tracklet_path):
                    os.symlink(tracklet_path, os.path.join(temp_dir, args.tracklet))
                    command = f"conda run -n {config.reid_env} python3 {config.reid_script} --tracklets_folder {temp_dir} --output_folder {features_dir}"
                    success = os.system(command) == 0
                else:
                    print(f"Tracklet {args.tracklet} not found.")
                    success = False
        else:
            command = f"conda run -n {config.reid_env} python3 {config.reid_script} --tracklets_folder {image_dir} --output_folder {features_dir}"
            success = os.system(command) == 0
        print("Done generating features")

    if args.pipeline['filter'] and success:
        print("Filtering outliers...")
        command = f"python3 gaussian_outliers.py --tracklets_folder {image_dir} --output_folder {features_dir}"
        success = os.system(command) == 0
        print("Done filtering")

    if args.pipeline['legible'] and success:
        print("Classifying legibility...")
        try:
            legible_dict, _ = get_soccer_net_legibility_results(args, use_filtered=True, filter='gauss', exclude_balls=False)
        except Exception as e:
            print(f"Error: {e}")
            success = False
        print("Done legibility classification")

    if args.pipeline['pose'] and success:
        print("Generating pose JSON...")
        generate_json_for_pose_estimator(args, legible_dict)
        print("Running pose estimation...")
        input_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                  config.dataset['SoccerNet'][args.part]['pose_input_json'])
        output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                   config.dataset['SoccerNet'][args.part]['pose_output_json'])
        command = f"conda run -n {config.pose_env} python3 pose.py {config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
            {config.pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} --out-json {output_json}"
        success = os.system(command) == 0
        print("Done pose estimation")

    if args.pipeline['crops'] and success:
        print("Generating crops...")
        crops_destination_dir = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['crops_folder'], 'imgs')
        Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
        helpers.generate_crops(
            os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['pose_output_json']),
            crops_destination_dir,
            legible_dict
        )
        print("Done generating crops")

    if args.pipeline['str'] and success:
        print("Running STR...")
        image_dir_crops = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['crops_folder'])
        str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['jersey_id_result'])
        command = f"conda run -n {config.str_env} python3 str.py {config.dataset['SoccerNet']['str_model']} --data_root={image_dir_crops} --batch_size=64 --inference --result_file {str_result_file}"
        success = os.system(command) == 0
        print("Done STR")

    if args.pipeline['combine'] and success:
        print("Combining results...")
        str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['jersey_id_result'])
        results_dict, _ = helpers.process_jersey_id_predictions(str_result_file, useBias=True)
        consolidated_dict = results_dict  # Simplified for single tracklet
        final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['final_result'])
        with open(final_results_path, 'w') as f:
            json.dump(consolidated_dict, f)
        print("Results saved")

    if args.pipeline['eval'] and success:
        print("Evaluating...")
        gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['gt'])
        with open(final_results_path, 'r') as f:
            consolidated_dict = json.load(f)
        with open(gt_path, 'r') as gf:
            gt_dict = json.load(gf)
        helpers.evaluate_results(consolidated_dict, gt_dict)
        print("Evaluation complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Options: 'SoccerNet'")
    parser.add_argument('part', help="Options: 'test', 'val', 'train', 'challenge'")
    parser.add_argument('--tracklet', type=str, help="Specific tracklet ID to process")
    parser.add_argument('--train_str', action='store_true', default=False, help="Train STR model")
    args = parser.parse_args()

    if args.dataset != 'SoccerNet':
        print("This script is modified for SoccerNet only.")
        exit(1)

    args.pipeline = {
        "soccer_ball_filter": True,
        "feat": True,
        "filter": True,
        "legible": True,
        "pose": True,
        "crops": True,
        "str": True,
        "combine": True,
        "eval": True
    }

    soccer_net_pipeline(args)
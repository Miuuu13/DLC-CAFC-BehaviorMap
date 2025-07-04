#%% IMPORTS

import os
import re
import random
from collections import defaultdict
from typing import List
import deeplabcut as dlc
import pandas as pd
import yaml
import json


#%%
CONFIG_PATH = "OCNC_DLC_BEH-MANUELA-2025-07-03/config.yaml"

#%%

def get_unique_videos_from_sessions(video_dirs: List[str], n: int = 1) -> List[str]:
    """
    Select `n` unique animal IDs per session (s1–s10).
    Total videos = 10 sessions × n.

    Parameters:
        video_dirs (List[str]): List of directories to search.
        n (int): Number of animals to select per session.

    Returns:
        List[str]: List of selected video paths.
    """
    session_pattern = re.compile(r'_(s\d+)_')
    animal_id_pattern = re.compile(r'^(\d+)_')

    videos_by_session = defaultdict(lambda: defaultdict(list))

    for video_dir in video_dirs:
        for fname in os.listdir(video_dir):
            if fname.endswith(".mp4"):
                session_match = session_pattern.search(fname)
                id_match = animal_id_pattern.match(fname)
                if session_match and id_match:
                    session = session_match.group(1)
                    animal_id = id_match.group(1)
                    full_path = os.path.join(video_dir, fname)
                    videos_by_session[session][animal_id].append(full_path)

    selected_videos = []

    for i in range(1, 11):  # sessions s1 to s10
        session_key = f's{i}'
        if session_key not in videos_by_session:
            continue

        animal_ids = list(videos_by_session[session_key].keys())
        random.shuffle(animal_ids)
        selected_ids = animal_ids[:n]

        for animal_id in selected_ids:
            selected_videos.append(videos_by_session[session_key][animal_id][0])  # first video for that ID

    return selected_videos

# === USER INPUTS ===
PROJECT_NAME = "OCNC_DLC_BEH"
EXPERIMENTER = "MANUELA"
VIDEO_DIRS = ["/home/manuela/Videos/DATA_BEH_Transferred_renamed_2025JUN30/ALL_Batch_A-F/"]
N = 2  # number of animals per session (total videos = N x 10)

# === SELECT VIDEOS & CREATE PROJECT ===
video_list = get_unique_videos_from_sessions(VIDEO_DIRS, n=N)
print(f"Selected {len(video_list)} videos:")
for v in video_list:
    print(v)


#only run if you want to create a project:
# dlc.create_new_project(PROJECT_NAME, EXPERIMENTER, video_list, copy_videos=True)

#%%


#%%

#%%
import os

cwd = os.getcwd()
print(cwd)

#%%

import cv2

# Load an image from file
# image = cv2.imread('/home/manuela/OCNC-DLC-2025/OCNC-MANUELA-2025-06-28/labeled-data/994_A_s9_rm_12_4kHz/img01263.png')
# print(image.__len__())
# # Display the image in a window
# cv2.imshow('Image Window', image)

# # Wait for a key press and close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #to get x1, x2 = 148 and y1,y2 = 482



#%%


CONFIG_PATH = "OCNC_DLC_BEH-MANUELA-2025-07-03/config.yaml"

import yaml


# with open(CONFIG_PATH, "r") as f:
#     cfg = yaml.safe_load(f)

# cfg["cropping"] = True
# cfg["crop"] = [152, 482, 150, 482]

# with open(CONFIG_PATH, "w") as f:
#     yaml.dump(cfg, f)

# print(" Cropping updated in config.yaml.")

#after this there 


#%%

import yaml

# to crop all videos same - check github, 2025JUL3-11pm if i want to crop each video differently


# with open(CONFIG_PATH, "r") as f:
#     cfg = yaml.safe_load(f)

# # Set global crop
# cfg["cropping"] = True
# cfg["crop"] = [152, 482, 150, 482]

# # Remove per-video crops
# for video in cfg["video_sets"]:
#     if isinstance(cfg["video_sets"][video], dict) and "crop" in cfg["video_sets"][video]:
#         del cfg["video_sets"][video]["crop"]

# with open(CONFIG_PATH, "w") as f:
#     yaml.dump(cfg, f)

# print(" Cleaned up config.yaml — global crop will now apply to all videos.")



#%%
import yaml

def update_config_yaml(config_path: str):
    # Custom bodypart names
    new_bodyparts = [
        "1_SNOUT",
        "2_LEFT_EAR",
        "3_RIGHT_EAR",
        "4_NECK",
        "5_LEFT_SHOULDER",
        "6_RIGHT_SHOULDER",
        "7_LEFT_HIP",
        "8_RIGHT_HIP",
        "9_TAIL_BASE",
        "10_TAIL_FIRST_THIRD",
        "11_TAIL_SECOND_THIRD",
        "12_TAIL_END"
    ]

    # Skeleton connections in top-down view
    new_skeleton = [
        ["1_SNOUT", "2_LEFT_EAR"],
        ["1_SNOUT", "3_RIGHT_EAR"],
        ["1_SNOUT", "4_NECK"],
        ["4_NECK", "5_LEFT_SHOULDER"],
        ["4_NECK", "6_RIGHT_SHOULDER"],
        ["5_LEFT_SHOULDER", "7_LEFT_HIP"],
        ["6_RIGHT_SHOULDER", "8_RIGHT_HIP"],
        ["7_LEFT_HIP", "9_TAIL_BASE"],
        ["8_RIGHT_HIP", "9_TAIL_BASE"],
        ["9_TAIL_BASE", "10_TAIL_FIRST_THIRD"],
        ["10_TAIL_FIRST_THIRD", "11_TAIL_SECOND_THIRD"],
        ["11_TAIL_SECOND_THIRD", "12_TAIL_END"]
    ]

    # Load the config.yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update bodyparts and skeleton
    config["bodyparts"] = new_bodyparts
    config["skeleton"] = new_skeleton

    # Remove deprecated fields if they exist
    for k in ["x1", "x2", "y1", "y2"]:
        config.pop(k, None)

    # Apply cropping info to each video
    for video_path in config.get("video_sets", {}):
        config["video_sets"][video_path]["crop"] = [152, 482, 150, 482]

    # Save the updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✔ config.yaml updated successfully at: {config_path}")
# Example usage:
#only run if update needed:
#update_config_yaml("/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/config.yaml")



#%%
# fix crop writing in .yaml
import deeplabcut as dlc

import yaml

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

for video in cfg['video_sets']:
    if isinstance(cfg['video_sets'][video], dict) and 'crop' in cfg['video_sets'][video]:
        crop_list = cfg['video_sets'][video]['crop']
        if isinstance(crop_list, list):
            cfg['video_sets'][video]['crop'] = ",".join(map(str, crop_list))

with open(CONFIG_PATH, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print("Fixed crop format in config.yaml.")



#%%% 

#NOTE cropping issue
#get rid of extracted frames and no crop!

import os
import shutil
import yaml

# === CONFIG ===
config_path = "OCNC_DLC_BEH-MANUELA-2025-07-03/config.yaml"
project_dir = os.path.dirname(config_path)

# === 1. DELETE LABELED-DATA FOLDERS ===
# labeled_data_path = os.path.join(project_dir, "labeled-data")
# if os.path.exists(labeled_data_path):
#     for folder in os.listdir(labeled_data_path):
#         full_path = os.path.join(labeled_data_path, folder)
#         if os.path.isdir(full_path):
#             shutil.rmtree(full_path)
#     print("Deleted all labeled-data folders.")
# else:
#     print("No labeled-data folder found.")

# # === 2. REMOVE CROPPING FROM CONFIG ===
# with open(config_path, "r") as f:
#     cfg = yaml.safe_load(f)

# # Remove global crop settings
# cfg.pop("cropping", None)
# cfg.pop("crop", None)

# # Remove per-video crop
# for video_path in cfg.get("video_sets", {}):
#     if "crop" in cfg["video_sets"][video_path]:
#         del cfg["video_sets"][video_path]["crop"]

# with open(config_path, "w") as f:
#     yaml.dump(cfg, f, default_flow_style=False)

# print("Cleaned crop info from config.yaml.")

#%%

"""reintegrate the cropping, but well done (not QAD, one does not fit all!)"""


# 
""" try to get coordinates """                                            

import cv2
import os
import json

# List of video paths (use your `video_list`)
video_list = [
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1002_B_s10_rp_12_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1002_B_s6_rp_12_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1012_B_s6_rm_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1020_C_s10_rm_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1020_C_s4_rm_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1023_F_s1_rm_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1031_B_s3_rp_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1032_F_s4_rp_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/1040_F_s2_rp_12_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/935_B_s5_rm_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/942_D_s9_rp_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/953_D_s2_rm_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/953_D_s7_rm_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/955_F_s9_rp_12_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/971_E_s7_rp_12_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/971_E_s8_rp_12_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/974_C_s8_rm_7_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/986_D_s1_rm_12_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/988_A_s5_rp_12_4kHz.mp4",
    "/home/manuela/Videos/OCNC_Project_DLC_BEH_2025JUL03/OCNC_DLC_BEH-MANUELA-2025-07-03/videos/990_B_s3_rm_12_4kHz.mp4"
]

# Output dictionary
crop_coords = {}

# Output path to save crop data
output_json = "arena_crops.json"

for video_path in video_list:
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)

    # Grab a frame (e.g., first frame)
    success, frame = cap.read()
    if not success:
        print(f"Could not read from video: {video_path}")
        cap.release()
        continue

    # Resize for easier viewing if too big (optional)
    scale = 2 #0.5
    resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Let user select ROI (returns x, y, w, h)
    r = cv2.selectROI(f"Select Arena - {os.path.basename(video_path)}", resized, showCrosshair=True)
    cv2.destroyAllWindows()

    # Scale coordinates back to original size
    x, y, w, h = [int(val / scale) for val in r]
    x1, x2 = x, x + w
    y1, y2 = y, y + h

    crop_coords[video_path] = [x1, x2, y1, y2]
    print(f"{os.path.basename(video_path)} -> crop: [{x1}, {x2}, {y1}, {y2}]")

    cap.release()

# Save to JSON
with open(output_json, "w") as f:
    json.dump(crop_coords, f, indent=2)

print(f"\nAll crop coordinates saved to {output_json}")


#%% 
""" did not work: remove later (fixed in the next cell)"""

# """ update the config file - add crooping using json file """

# def update_config_with_crop_json(config_path: str, json_path: str):
#     """
#     Update per-video cropping values in the config.yaml using a provided JSON file.

#     Args:
#         config_path (str): Path to the DLC config.yaml file.
#         json_path (str): Path to JSON file with cropping info (video_path -> [x1, x2, y1, y2]).
#     """
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")
#     if not os.path.exists(json_path):
#         raise FileNotFoundError(f"JSON file not found: {json_path}")

#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     with open(json_path, 'r') as f:
#         crop_data = json.load(f)

#     config['cropping'] = True
#     config.pop('crop', None)  # Remove global crop if present

#     updated = 0
#     for video_path, coords in crop_data.items():
#         if video_path in config.get("video_sets", {}):
#             config["video_sets"][video_path]["crop"] = coords
#             updated += 1
#         else:
#             print(f"⚠ Skipped (not in config): {video_path}")

#     with open(config_path, 'w') as f:
#         yaml.dump(config, f, default_flow_style=False)

#     print(f"✔ Updated {updated} video entries with crop values in {os.path.basename(config_path)}.")

# #%% ex us
#     #only run if you wnat to overwrite config

#     # Use current working directory if needed:
# cwd = os.getcwd()
# json_path = os.path.join(cwd, "arena_crops.json")
# config_path = os.path.join(cwd, "OCNC_DLC_BEH-MANUELA-2025-07-03/config.yaml")

# update_config_with_crop_json(config_path, json_path)

#%%

#new try

def update_config_with_crop_json(config_path: str, json_path: str):
    import yaml
    import json

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(json_path, 'r') as f:
        crop_data = json.load(f)

    config['cropping'] = True
    config.pop('crop', None)  # Remove global crop if present

    updated = 0
    for video_path, coords in crop_data.items():
        if video_path in config.get("video_sets", {}):
            # Convert list [x1, x2, y1, y2] → "x1,x2,y1,y2"
            crop_str = ",".join(map(str, coords))
            config["video_sets"][video_path]["crop"] = crop_str
            updated += 1
        else:
            print(f"⚠ Skipped (not in config): {video_path}")

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✔ Updated {updated} video entries with crop values in {os.path.basename(config_path)}.")

update_config_with_crop_json(config_path, json_path)

#%%
""" EXTRACT FRAMES -  with specific cropping per video """
# NOTE: Choose the method that allows you to select the frames that capture all the behaviour you are interested in :)
#%%
CONFIG_PATH = "OCNC_DLC_BEH-MANUELA-2025-07-03/config.yaml"
#%%
dlc.extract_frames(
    CONFIG_PATH,
    mode='automatic',
    algo='kmeans', #uniform is default?
    userfeedback=False,
    crop=True #True  # <-- this enforces the crop during extraction
)

#TODO care about crop later

#%%
# for opening GUI, labeling:

dlc.label_frames(CONFIG_PATH)







#%%


#%% CHECK ASSIGNMENT
dlc.check_labels(CONFIG_PATH, visualizeindividuals=False)


# %%
dlc.create_training_dataset(CONFIG_PATH)
# %%to train network
dlc.train_network(CONFIG_PATH)
# %%
DATA_DIR = "OCNC-MANUELA-2025-06-28/training-datasets/iteration-0/UnaugmentedDataSet_OCNCJun28/CollectedData_MANUELA.h5"
df = pd.read_hdf(DATA_DIR+"M_190124_110324_12_60fpsDLC_resnet50_OCNCJun19shuffle1_15000.h5")
df.head()

#%%

# Enter the list of videos to analyze.
VIDEO_PATH = ["/home/manuela/Videos/OCNC-PROJECT_DLC_2025JUN28/Batch_A_2022/936_A_s6_rp_7_4kHz.mp4"]
dlc.analyze_videos(CONFIG_PATH, VIDEO_PATH, videotype=".mp4")












# ############################

#%% IMPORTS

import os
import re
from collections import defaultdict
from typing import List
import deeplabcut as dlc


#%%

import os
import re
from collections import defaultdict
from typing import List

def get_video_paths_by_session_unique_ids(video_dirs: List[str], per_session_n_ids: int = 10, videos_per_id: int = 1) -> List[str]:
    """
    Get up to `videos_per_id` videos from `per_session_n_ids` unique animal IDs per session (s1–s10).

    Parameters:
        video_dirs (List[str]): List of directories to search.
        per_session_n_ids (int): Number of unique animal IDs per session.
        videos_per_id (int): Number of videos to select per animal ID.

    Returns:
        List[str]: List of selected video paths.
    """
    session_pattern = re.compile(r'_(s\d+)_')
    animal_id_pattern = re.compile(r'^(\d+)_')  # animal ID is at the start of the filename

    # Store: {session -> {animal_id -> [videos]}}
    videos_by_session_and_id = defaultdict(lambda: defaultdict(list))

    for video_dir in video_dirs:
        for file in os.listdir(video_dir):
            if file.endswith('.mp4'):
                session_match = session_pattern.search(file)
                id_match = animal_id_pattern.match(file)
                if session_match and id_match:
                    session = session_match.group(1)
                    animal_id = id_match.group(1)
                    full_path = os.path.join(video_dir, file)
                    videos_by_session_and_id[session][animal_id].append(full_path)

    selected_videos = []

    for i in range(1, 11):  # sessions s1 to s10
        session_key = f's{i}'
        animals = list(videos_by_session_and_id[session_key].keys())
        animals = sorted(animals)[:per_session_n_ids]  # take first sN IDs alphabetically (you can randomize this if preferred)

        for animal_id in animals:
            videos = videos_by_session_and_id[session_key][animal_id][:videos_per_id]
            selected_videos.extend(videos)

    return selected_videos

#%%
#see example function call + output
VIDEO_DIRS = ["/home/manuela/Videos/DATA_BEH_Transferred_renamed_2025JUN30/ALL_Batch_A-F/"]
video_list = get_video_paths_by_session_unique_ids(VIDEO_DIRS, per_session_n_ids=10, videos_per_id=1)
print(video_list)


#%%

""" ==== CREATE PROJECT ==== """

# -- edit this part to match your directory --
# -- input: str 
PROJECT_NAME: str = "OCNC_DLC_BEH"
EXPERIMENTER: str = "MANUELA"
VIDEO_DIRS = ["/home/manuela/Videos/DATA_BEH_Transferred_renamed_2025JUN30/ALL_Batch_A-F/"]
# WORKING_DIR: str = "models/"


# VIDEO SELECTION

#old: 
# NOTE: the VIDEO_DIR is a lst input -- you can use 
#dlc.create_new_project(PROJECT_NAME, EXPERIMENTER, [VIDEO_DIR+VIDEO_NAME] #[VIDEO_DIR+"994_A_s9_rm_12.4kHz.mp4"]
#)

#new:
video_list = get_video_paths_by_session_unique_ids(VIDEO_DIRS, per_session_n_ids=10, videos_per_id=2)

print(f"Total selected videos: {len(video_list)}")
print(video_list)  # Preview

#%% CREATE DLC PROJECT

dlc.create_new_project(PROJECT_NAME, EXPERIMENTER, video_list, copy_videos=True)




os



#%%



#%% DEFINE PARAMETERS

# Project metadata
PROJECT_NAME = "OCNC_DLC_BEH"
EXPERIMENTER = "MANUELA"

# Source videos (full list with all batches)
VIDEO_DIRS = ["/home/manuela/Videos/DATA_BEH_Transferred_renamed_2025JUN30/ALL_Batch_A-F/"]

# Select videos (10 IDs per session, 2 videos per ID)
video_list = get_video_paths_by_session_unique_ids(VIDEO_DIRS, per_session_n_ids=10, videos_per_id=2)

# Check
print(f"Total videos selected: {len(video_list)}")
print(video_list[:5])  # preview

#%% CREATE PROJECT

# This replaces the old VIDEO_DIR + VIDEO_NAME logic
dlc.create_new_project(PROJECT_NAME, EXPERIMENTER, video_list, copy_videos=True)

import numpy as np
from collections import defaultdict
import json
from utils import myprint
import os
import argparse
from PIL import Image
import random
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # This prevents JAX from preallocating most of the GPU memory.

def get_ep_idx_to_info(total_episodes):
	# constants
	ds_name = "droid_new"
	ds_fol = f"{ds_name}_broken_up"
	all_objects = ["marker", "cloth", "bottle", "block", "drawer", "lid", "mug", "cup"]
	# read classifications.csv to get scene_id to location_name
	scene_id_to_location_name = {}
	with open(f"droid_info/classifications.csv", "r") as csv_file:
		lines = csv_file.readlines()
		for line in lines[1:]:
			scene_id, location_name = line.strip().split(",")
			scene_id_to_location_name[int(scene_id)] = location_name.strip('\n')
	# here is where we get the ep_idx_to_info dict
	ep_idx_to_info = {}
	for ep_idx in range(total_episodes):
		# read metadata; continue if no language instructions
		with open(f"{ds_fol}/episode_{ep_idx}.json", "r") as json_file:
			ep_metadata = json.load(json_file)
		if ep_metadata["language_instruction"] == "" and ep_metadata["language_instruction_2"] == "" and ep_metadata["language_instruction_3"] == "":
			continue
		# assign object name in the all_objects list or continue if not found
		assigned_object_name = False
		for obj_name in all_objects:
			if obj_name in ep_metadata["language_instruction"] or obj_name in ep_metadata["language_instruction_2"] or obj_name in ep_metadata["language_instruction_3"]:
				ep_idx_to_info[ep_idx] = {"object_name": obj_name}
				assigned_object_name = True
				break # one object name per episode
		if not assigned_object_name:
			continue
		# assign metadata
		ep_idx_to_info[ep_idx].update(ep_metadata)
		# assign num_steps
		ep_idx_to_info[ep_idx]["num_steps"] = ep_metadata["shapes"]["observation__wrist_image_left"][0]
		# assign location_name
		ep_idx_to_info[ep_idx]["location_name"] = scene_id_to_location_name[ep_metadata["scene_id"]] if ep_metadata["scene_id"] in scene_id_to_location_name else "Unknown"
	return ep_idx_to_info

def get_chosen_id_to_ep_idxs(chosen_id, ep_idx_to_info):
	chosen_id_to_ep_idxs = defaultdict(set)
	for ep_idx, ep_metadata in ep_idx_to_info.items():
		if chosen_id == "location_name": # goes from larger groups to smaller groups as you go down this if-else chain
			key = ep_metadata["location_name"]
		elif chosen_id == "object_name":
			key = ep_metadata["object_name"]
		elif chosen_id == "scene_id":
			key = ep_metadata["scene_id"]
		elif chosen_id == "scene_id_and_object_name":
			key = (ep_metadata["scene_id"], ep_metadata["object_name"])
		elif chosen_id == "scene_id_and_object_name_and_task_category":
			key = (ep_metadata["scene_id"], ep_metadata["object_name"], ep_metadata["task_category"])
		else:
			raise NotImplementedError(f'{chosen_id=} is not valid')
		chosen_id_to_ep_idxs[key].add(ep_idx)
	chosen_id_to_ep_idxs = {chosen_id: list(ep_idxs) for chosen_id, ep_idxs in chosen_id_to_ep_idxs.items()} # convert sets to lists
	return chosen_id_to_ep_idxs

def as_gif(images, path="temp.gif"):
	# convert to PIL images
	images = [Image.fromarray(image) for image in images]
	# Render the images as the gif (15Hz control frequency):
	myprint(f'converted to PIL Images; creating gif')
	images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000/15), loop=0)
	# gif_bytes = open(path,"rb").read()
	# return gif_bytes

def add_text_overlay(frame, lang_1, lang_2, lang_3):
    # Define font and text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_color = (255, 255, 255)  # White color
    thickness = 1
    line_type = cv2.LINE_AA

    # Define text positions
    positions = [(10, 20), (10, 40), (10, 60)]  # (x, y) coordinates
    texts = [lang_1, lang_2, lang_3]

    # Overlay text on frame
    for text, pos in zip(texts, positions):
        cv2.putText(frame, text, pos, font, font_scale, font_color, thickness, line_type)

    return frame

def group_by_chosen_id(chosen_id, only_count, total_episodes, min_num_episodes, N, M):	
	ep_idx_to_info = get_ep_idx_to_info(total_episodes)
	chosen_id_to_ep_idxs = get_chosen_id_to_ep_idxs(chosen_id, ep_idx_to_info)
	myprint(f'got the info dicts: there are {len(ep_idx_to_info)} groupings\n')

	# create histogram of num_ep_idxs in chosen_id_to_ep_idxs
	chosen_id_to_ep_idxs_with_atleast_min_num_episodes = {chosen_id: ep_idxs for chosen_id, ep_idxs in chosen_id_to_ep_idxs.items() if len(ep_idxs) >= min_num_episodes}
	myprint(f'number of groupings with atleast {min_num_episodes} episodes [in the first {total_episodes} episodes]: {len(chosen_id_to_ep_idxs_with_atleast_min_num_episodes)}\n')
	print('these groupings have chosen_id-->num_episodes as\n\n', {chosen_id: len(ep_idxs) for chosen_id, ep_idxs in chosen_id_to_ep_idxs_with_atleast_min_num_episodes.items()})
	if only_count:
		return

	# randomly sample N chosen_ids and randomly sample M ep_idxs from each chosen_id
	assert M <= min_num_episodes
	sampled_chosen_ids = sorted(random.sample(list(chosen_id_to_ep_idxs_with_atleast_min_num_episodes.keys()), N))
	myprint(f'{sampled_chosen_ids=}') # (N,)
	sampled_ep_idxs_2D_list = [sorted(random.sample(chosen_id_to_ep_idxs_with_atleast_min_num_episodes[chosen_id], M)) for chosen_id in sampled_chosen_ids]
	myprint(f'{sampled_ep_idxs_2D_list=}') # (N, M)
	assert np.array(sampled_ep_idxs_2D_list).shape == (N, M)

	# get max_num_steps in sampled_ep_idxs_2D_list
	max_num_steps = max([ep_idx_to_info[ep_idx]["num_steps"] for row_of_ep_idxs in sampled_ep_idxs_2D_list for ep_idx in row_of_ep_idxs])
	myprint(f'{max_num_steps=}')

	# load videos for each episode
	ds_name = "droid_new"
	ds_fol = f"{ds_name}_broken_up"
	ep_idx_to_video = {}
	for i in range(N):
		for j in range(M):
			ep_idx = sampled_ep_idxs_2D_list[i][j]
			with open(f"{ds_fol}/episode_{ep_idx}.json", "r") as json_file:
				ep_metadata = json.load(json_file)
			ep_idx_to_info[ep_idx] = {"language_instruction": ep_metadata["language_instruction"],
										"language_instruction_2": ep_metadata["language_instruction_2"],
										"language_instruction_3": ep_metadata["language_instruction_3"]}
			ep_steps = np.load(f"{ds_fol}/episode_{ep_idx}.npz")
			ep_video = ep_steps["observation__wrist_image_left"]
			if len(ep_video) < max_num_steps:
				ep_video = np.concatenate([ep_video] + [deepcopy(ep_video[-1:]) for _ in range(max_num_steps - len(ep_video))])
			if ep_idx in ep_idx_to_video:
				myprint(f'ep_idx {ep_idx} already exists')
			ep_idx_to_video[ep_idx] = ep_video
	assert len(ep_idx_to_video) == N * M, f'{len(ep_idx_to_video)=} != {N * M=}'
	myprint(f'collected videos')

	# now create a list of max_num_steps length, where each element is a big frame of shape (N * H, M * W, 3)
	big_frame_list = []
	for step_idx in range(max_num_steps):
		big_frame = []
		for i in range(N):
			big_row = []
			for j in range(M):
				ep_idx = sampled_ep_idxs_2D_list[i][j]
				frame = ep_idx_to_video[ep_idx][step_idx]
				lang_1, lang_2, lang_3 = ep_idx_to_info[ep_idx]["language_instruction"], ep_idx_to_info[ep_idx]["language_instruction_2"], ep_idx_to_info[ep_idx]["language_instruction_3"]
				frame = add_text_overlay(frame, lang_1, lang_2, lang_3)
				big_row.append(frame)
			big_frame.append(np.concatenate(big_row, axis=1))
		big_frame_list.append(np.concatenate(big_frame, axis=0))
	myprint(f'collected big frames')
	
	# make gif
	os.makedirs("droid_groups", exist_ok=True)
	save_path = f"droid_groups/{ds_name}_N{N}_M{M}_{chosen_id}_withlang_below{total_episodes // 1000}k.gif"
	as_gif(big_frame_list, path=save_path)
	myprint(f'saved gif to {save_path}')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--chosen_id", type=str, default="scene_id_and_object_name", choices=["location_name", "object_name", "scene_id", "scene_id_and_object_name", "scene_id_and_object_name_and_task_category"])
	parser.add_argument("--only_count", action="store_true")
	parser.add_argument("--total_episodes", type=int, default=95658)
	parser.add_argument("--min_num_episodes_in_each_grouping", type=int, default=10)
	parser.add_argument("--N", type=int, default=10, help="number of groupings to sample for the gif visualization")
	parser.add_argument("--M", type=int, default=5, help="number of episodes to sample from each grouping for the gif visualization")
	args = parser.parse_args()
	group_by_chosen_id(args.chosen_id, args.only_count, args.total_episodes, args.min_num_episodes_in_each_grouping, args.N, args.M)
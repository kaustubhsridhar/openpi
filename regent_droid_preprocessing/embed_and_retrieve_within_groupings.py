import numpy as np
from collections import defaultdict
import json
from utils import myprint
import os
import argparse
from openpi.shared.image_tools import resize_with_pad
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

def embed_images_with_pi0(batch, siglip):
	assert len(batch.shape) == 4 and batch.shape[1] == 224 and batch.shape[2] == 224 and batch.shape[3] == 3, f'{batch.shape=}'
	assert batch.dtype == np.uint8

	batch = batch.astype(np.float32) / 255.0 * 2.0 - 1.0 # normalize to [-1, 1]

	features = siglip(batch, train=False)[0] # (batch_size, num_tokens, 2048) # conversion to jax happens inside this function
	features = features.mean(axis=1) # (batch_size, 2048)
	features = np.array(features).astype(np.float32)

	return features

def group_by_chosen_id(chosen_id, total_episodes, min_num_episodes):	
	ep_idx_to_info = get_ep_idx_to_info(total_episodes)
	chosen_id_to_ep_idxs = get_chosen_id_to_ep_idxs(chosen_id, ep_idx_to_info)
	myprint(f'got the info dicts: there are {len(ep_idx_to_info)} groupings\n')

	# create histogram of num_ep_idxs in chosen_id_to_ep_idxs
	chosen_id_to_ep_idxs_with_atleast_min_num_episodes = {chosen_id: ep_idxs for chosen_id, ep_idxs in chosen_id_to_ep_idxs.items() if len(ep_idxs) >= min_num_episodes}
	myprint(f'number of groupings with atleast {min_num_episodes} episodes [in the first {total_episodes} episodes]: {len(chosen_id_to_ep_idxs_with_atleast_min_num_episodes)}\n')
	print('these groupings have chosen_id-->num_episodes as\n\n', {chosen_id: len(ep_idxs) for chosen_id, ep_idxs in chosen_id_to_ep_idxs_with_atleast_min_num_episodes.items()})

	return chosen_id_to_ep_idxs_with_atleast_min_num_episodes

def embed_episodes(chosen_id_to_ep_idxs_with_atleast_min_num_episodes):
	for chosen_id_count, (chosen_id, ep_idxs) in enumerate(chosen_id_to_ep_idxs_with_atleast_min_num_episodes.items()):
		for ep_count, ep_idx in enumerate(ep_idxs):
			if not os.path.exists(f"{ds_emb_fol}/episode_{ep_idx}.npz"):
				# embed the three videos in the episode
				# read
				steps = np.load(f"{ds_fol}/episode_{ep_idx}.npz")
				observation__exterior_image_1_left = steps["observation__exterior_image_1_left"] # (num_steps, 180, 320, 3)
				observation__exterior_image_2_left = steps["observation__exterior_image_2_left"] # (num_steps, 180, 320, 3)
				observation__wrist_image_left = steps["observation__wrist_image_left"] # (num_steps, 180, 320, 3)
				assert observation__exterior_image_1_left.dtype == observation__exterior_image_2_left.dtype == observation__wrist_image_left.dtype == np.uint8
				# resize_with_pad to 224x224
				observation__exterior_image_1_left = resize_with_pad(observation__exterior_image_1_left, 224, 224)
				observation__exterior_image_2_left = resize_with_pad(observation__exterior_image_2_left, 224, 224)
				observation__wrist_image_left = resize_with_pad(observation__wrist_image_left, 224, 224)
				# embed
				embeddings__exterior_image_1_left = embed_images_with_pi0(observation__exterior_image_1_left, pi0)
				embeddings__exterior_image_2_left = embed_images_with_pi0(observation__exterior_image_2_left, pi0)
				embeddings__wrist_image_left = embed_images_with_pi0(observation__wrist_image_left, pi0)
				# save
				np.savez(f"{ds_emb_fol}/episode_{ep_idx}.npz", 
			 				embeddings__exterior_image_1_left=embeddings__exterior_image_1_left, 
							embeddings__exterior_image_2_left=embeddings__exterior_image_2_left, 
							embeddings__wrist_image_left=embeddings__wrist_image_left)
				myprint(f'embedded episode {ep_idx} [episode count {ep_count}/{len(ep_idxs)}]')
			else:
				myprint(f'skipping episode {ep_idx} [episode count {ep_count}/{len(ep_idxs)}]')
		myprint(f'finished embedding all episodes for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')
	myprint(f'done embedding!')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--chosen_id", type=str, default="scene_id_and_object_name", choices=["location_name", "object_name", "scene_id", "scene_id_and_object_name", "scene_id_and_object_name_and_task_category"])
	parser.add_argument("--total_episodes", type=int, default=95658)
	parser.add_argument("--min_num_episodes_in_each_grouping", type=int, default=10)
	args = parser.parse_args()

	# setup
	chosen_id_to_ep_idxs_with_atleast_min_num_episodes = group_by_chosen_id(args.chosen_id, args.total_episodes, args.min_num_episodes_in_each_grouping)
	
	ds_name = "droid_new"
	ds_fol = f"{ds_name}_broken_up"
	ds_emb_fol = f"{ds_name}_broken_up_embeddings"
	num_groupings = len(chosen_id_to_ep_idxs_with_atleast_min_num_episodes)

	# model
	pi0 = 
	
	embed_episodes(chosen_id_to_ep_idxs_with_atleast_min_num_episodes)
	
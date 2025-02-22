import os
import tensorflow_datasets as tfds
import numpy as np
import json
import argparse
from utils import myprint
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # This prevents JAX from preallocating most of the GPU memory.

# args
parser = argparse.ArgumentParser()
parser.add_argument("--start_episode", type=int, default=0)
args = parser.parse_args()

# Dataset name and output folder
ds_name = "droid_new"
new_fol_name = f"{ds_name}_broken_up"
os.makedirs(new_fol_name, exist_ok=True)

# Load dataset
ds = tfds.load(ds_name, data_dir="./", split="train")
tot = len(ds)

# Break up the dataset into indivudal episodes and save them in the new folder
for ep_idx, ep in enumerate(ds):
	if ep_idx < args.start_episode - 3:
		myprint(f'skipped episode {ep_idx} / {tot}')
		continue
	elif args.start_episode - 10 <= ep_idx < args.start_episode: # for only some episodes before the start_episode...
		# check if the file_path in the ep["episode_metadata"] matches that in the saved json file
		existing_json_file = json.load(open(f"{new_fol_name}/episode_{ep_idx}.json", "r"))
		file_path_in_episode_metadata = ep["episode_metadata"]["file_path"].numpy().decode("utf-8")
		assert existing_json_file["file_path"] == file_path_in_episode_metadata, f"{existing_json_file['file_path']} != {file_path_in_episode_metadata}"
		myprint(f'verified and skipped episode {ep_idx} / {tot}')
		continue

	myprint(f"breaking off episode {ep_idx} / {tot}")
	metadata = {
		"building": ep["episode_metadata"]["building"].numpy().decode("utf-8"),
		"collector_id": ep["episode_metadata"]["collector_id"].numpy().decode("utf-8"),
		"date": ep["episode_metadata"]["date"].numpy().decode("utf-8"),
		"extrinsics_exterior_cam_1": ep["episode_metadata"]["extrinsics_exterior_cam_1"].numpy().tolist(),
		"extrinsics_exterior_cam_2": ep["episode_metadata"]["extrinsics_exterior_cam_2"].numpy().tolist(),
		"extrinsics_wrist_cam": ep["episode_metadata"]["extrinsics_wrist_cam"].numpy().tolist(),
		"file_path": ep["episode_metadata"]["file_path"].numpy().decode("utf-8"),
		"recording_folderpath": ep["episode_metadata"]["recording_folderpath"].numpy().decode("utf-8"),
		"scene_id": int(ep["episode_metadata"]["scene_id"].numpy()),
		"task_category": ep["episode_metadata"]["task_category"].numpy().decode("utf-8"),
	}
	steps = {
		"action": [],
		"action_dict__cartesian_position": [],
		"action_dict__cartesian_velocity": [],
		"action_dict__gripper_position": [],
		"action_dict__gripper_velocity": [],
		"action_dict__joint_position": [],
		"action_dict__joint_velocity": [],
		"discount": [],
		"is_first": [],
		"is_last": [],
		"is_terminal": [],
		"observation__cartesian_position": [],
		"observation__exterior_image_1_left": [],
		"observation__exterior_image_2_left": [],
		"observation__gripper_position": [],
		"observation__joint_position": [],
		"observation__wrist_image_left": [],
		"reward": [],
	}
	for step_idx, step in enumerate(ep["steps"]):
		steps["action"].append(step["action"].numpy().astype(np.float32)) # og: float64
		steps["action_dict__cartesian_position"].append(step["action_dict"]["cartesian_position"].numpy().astype(np.float32)) # og: float64
		steps["action_dict__cartesian_velocity"].append(step["action_dict"]["cartesian_velocity"].numpy().astype(np.float32)) # og: float64
		steps["action_dict__gripper_position"].append(step["action_dict"]["gripper_position"].numpy().astype(np.float32)) # og: float64
		steps["action_dict__gripper_velocity"].append(step["action_dict"]["gripper_velocity"].numpy().astype(np.float32)) # og: float64
		steps["action_dict__joint_position"].append(step["action_dict"]["joint_position"].numpy().astype(np.float32)) # og: float64
		steps["action_dict__joint_velocity"].append(step["action_dict"]["joint_velocity"].numpy().astype(np.float32)) # og: float64
		steps["discount"].append(step["discount"].numpy()) # og: float32
		steps["is_first"].append(step["is_first"].numpy()) # og: bool
		steps["is_last"].append(step["is_last"].numpy()) # og: bool
		steps["is_terminal"].append(step["is_terminal"].numpy()) # og: bool
		if step_idx == 0:
			metadata["language_instruction"] = step["language_instruction"].numpy().decode("utf-8")
			metadata["language_instruction_2"] = step["language_instruction_2"].numpy().decode("utf-8")
			metadata["language_instruction_3"] = step["language_instruction_3"].numpy().decode("utf-8")
		steps["observation__cartesian_position"].append(step["observation"]["cartesian_position"].numpy().astype(np.float32)) # og: float64
		steps["observation__exterior_image_1_left"].append(step["observation"]["exterior_image_1_left"].numpy()) # og: uint8
		steps["observation__exterior_image_2_left"].append(step["observation"]["exterior_image_2_left"].numpy()) # og: uint8
		steps["observation__gripper_position"].append(step["observation"]["gripper_position"].numpy().astype(np.float32)) # og: float64
		steps["observation__joint_position"].append(step["observation"]["joint_position"].numpy().astype(np.float32)) # og: float64
		steps["observation__wrist_image_left"].append(step["observation"]["wrist_image_left"].numpy()) # og: uint8
		steps["reward"].append(step["reward"].numpy()) # og: float32

	# stack all
	steps = {k: np.stack(v) for k, v in steps.items()}
	
	# add shapes of everything and dtype to metadata
	metadata["shapes"] = {k: list(v.shape) for k, v in steps.items()}
	metadata["dtypes"] = {k: str(v.dtype) for k, v in steps.items()}
		
	# save metadata to json and steps to npz
	with open(f"{new_fol_name}/episode_{ep_idx}.json", "w") as json_file:
		json.dump(metadata, json_file, indent=4)
	np.savez(f"{new_fol_name}/episode_{ep_idx}.npz", **steps)

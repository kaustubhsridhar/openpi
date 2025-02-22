import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from collections import defaultdict
import json
import jax
import jax.numpy as jnp
from tqdm import tqdm
from utils import myprint
import random
import os
import io
import cv2
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # This prevents JAX from preallocating most of the GPU memory.

def check_if_object_in_droid(object_name, total_episodes):
	# constants
	ds_name = "droid_new"
	ds_fol = f"{ds_name}_broken_up"
	# here is where we check
	for ep_idx in range(total_episodes):
		if os.path.exists(f"{ds_fol}/episode_{ep_idx}.json"):
			# read metadata; continue if no language instructions
			with open(f"{ds_fol}/episode_{ep_idx}.json", "r") as json_file:
				ep_metadata = json.load(json_file)
			if ep_metadata["language_instruction"] == "" and ep_metadata["language_instruction_2"] == "" and ep_metadata["language_instruction_3"] == "":
				continue
			# check if object name in language instructions
			object_name_exists = False
			if object_name in ep_metadata["language_instruction"] or object_name in ep_metadata["language_instruction_2"] or object_name in ep_metadata["language_instruction_3"]:
				object_name_exists = True
				break # one object name per episode
		else:
			print()
			myprint(f'{ds_fol}/episode_{ep_idx}.json does not exist; ending search at {ep_idx-1} instead of going until {total_episodes}')
			break
	return object_name_exists


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--object_name", type=str, default="marker")
	parser.add_argument("--total_episodes", type=int, default=95658)
	args = parser.parse_args()
	print(f'\nDoes {args.object_name} exist in droid in the first {args.total_episodes} episodes? {check_if_object_in_droid(args.object_name, args.total_episodes)}')

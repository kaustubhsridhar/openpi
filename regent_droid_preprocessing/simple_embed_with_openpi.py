from openpi.training import config
from openpi.policies import policy_config, droid_policy
from openpi.shared import download
import json
import numpy as np
import os
import time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # This prevents JAX from preallocating most of the GPU memory.

# start
config = config.get_config("pi0_fast_droid")
print(f'{config=}\n')
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print(f'{policy=}\n')

## uncomment below to see the policy load on your gpu vram
# time.sleep(1000)
# exit()

# Select an ep_idx and step_idx
ep_idx = 0
step_idx = 0

# Load the episode's metadata and steps
ds_name = "droid_new"
ds_fol = f"{ds_name}_broken_up"
with open(f"{ds_fol}/episode_{ep_idx}.json", "r") as json_file:
    metadata = json.load(json_file)
steps = np.load(f"{ds_fol}/episode_{ep_idx}.npz")
num_steps = steps["observation__exterior_image_1_left"].shape[0]

# Run inference on an example.
example = {
    "observation/exterior_image_1_left": steps["observation__exterior_image_1_left"][step_idx],
    # "observation/exterior_image_2_left": steps["observation__exterior_image_2_left"][step_idx],
    "observation/wrist_image_left": steps["observation__wrist_image_left"][step_idx],
    "observation/joint_position": steps["observation__joint_position"][step_idx],
    "observation/gripper_position": steps["observation__gripper_position"][step_idx],
    "prompt": metadata["language_instruction"]
}
## uncomment below to see an approved example
# example = droid_policy.make_droid_example()

# get output embedding only
embedding = policy.embed_only(example)
print(f'{embedding=}\n')

# get output action_chunk
action_chunk = policy.infer(example)["actions"]
print(f'{action_chunk=}\n')

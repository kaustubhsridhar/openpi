import jax
import os
from datetime import datetime
import numpy as np
from PIL import Image

def get_time():
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def myprint(s):
	print(f'{get_time()}: {s}')
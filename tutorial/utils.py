# importing the dependencies
import argparse
import numpy as np
import tensorflow as tf

def positional_encoding(self, pos):
	# @TODO: optimise this piece of code
	pos_embed = []
	for i in range(self.D_MODEL):
		angle = pos/(10**(8*i/self.D_MODEL))
		if i%2 == 0:
			pos_embed.append(np.sin(angle))
			continue
		pos_embed.append(np.cos(angle))
	return np.array(pos_embed, dtype = np.float32)

def load_data(filename):
	return pickle.load(open(filename, 'rb'))
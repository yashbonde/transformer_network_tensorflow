# importing the dependencies
import numpy as np

def positional_encoding(pos, e_dim):
	# @TODO: optimise this piece of code
	pos_embed = []
	for i in range(e_dim):
		angle = pos/(10**(4*i/e_dim))
		if i%2 == 0:
			pos_embed.append(np.sin(angle))
			continue
		pos_embed.append(np.cos(angle))
	return np.array(pos_embed, dtype = np.float32)

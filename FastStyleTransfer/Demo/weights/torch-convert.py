import sys
import torch
import numpy as np

# Usage:
# python torch-convert.py model.pth model
# (produces model.npz)

if __name__ == "__main__":
	in_file, out_file= sys.argv[1], sys.argv[2]
	state_dict = torch.load(in_file)
	npz = {}
	for label, tensor in state_dict.items():
		npz[label] = tensor.numpy()
	np.savez(out_file, **npz)

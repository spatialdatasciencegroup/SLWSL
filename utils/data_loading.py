import logging
from torch.utils.data import Dataset
import torch


class FormulaDataset(Dataset):
	def __init__(self, formulas, weights):
		assert len(formulas) == len(weights)
		self.formulas = formulas
		self.weights = weights

	def __getitem__(self, index):
		return self.formulas[index][0], self.formulas[index][1], self.weights[index]

	def __len__(self):
		return len(self.formulas)


class ImgDataset(Dataset):
	def __init__(self, image: torch.tensor, mask: torch.tensor, observe: torch.tensor, patch_size, stride, N=1):
		self.image_patches = image.data.unfold(0, 3, 3).unfold(1, patch_size, stride).unfold(2, patch_size, stride)


		self.mask_patches = mask.data.unfold(0, 1, 1).unfold(1, patch_size // N, stride // N).unfold(2, patch_size // N,
																									 stride // N)

		self.ob_patches = observe.data.unfold(0, 1, 1).unfold(1, patch_size // 100, stride // 100).unfold(2,
																										  patch_size // 100,
																										  stride // 100)

		self.dataset_size = self.mask_patches.shape[1] * self.mask_patches.shape[2]
		assert self.dataset_size == self.image_patches.shape[1] * self.image_patches.shape[
			2], 'Different image and mask numbers'

		logging.info(f'Creating dataset with {self.dataset_size} examples')

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, idx):
		x = idx // self.image_patches.shape[2]
		y = idx % self.image_patches.shape[2]
		return {
			'image': self.image_patches[0][x][y],
			'mask': self.mask_patches[0][x][y],
			'observe': self.ob_patches[0][x][y]
		}

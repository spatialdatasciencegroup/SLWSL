import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from utils.data_loading import ImgDataset
from utils.utils import EarlyStopping


def get_annotator_value(theta, observed, hidden):
	N = hidden.shape[-1] // observed.shape[-1]
	pooling = nn.AvgPool2d(N, N)
	hidden = pooling(hidden) > 0.5

	# 0,0 -> 0;
	# 0,1 -> 1;
	# 1,0 -> 2;
	# 1,1 -> 3
	res = []
	for i in range(hidden.shape[1]):
		for j in range(hidden.shape[2]):
			for k in range(hidden.shape[0]):
				temp = (observed[k][0][i][j], hidden[k][0][i][j])
				if temp == (0, 0):
					res.append(0)
				elif temp == (0, 1):
					res.append(1)
				elif temp == (1, 0):
					res.append(2)
				else:
					res.append(3)

	return theta(torch.tensor(res).to(hidden.device)).mean() * 1e-3


def train_model(
		model,
		device,
		image,
		mask,
		N,
		patch_size,
		stride,
		dataset_id,
		epochs: int = 5,
		batch_size: int = 1,
		learning_rate: float = 1e-5,
		weight_decay: float = 1e-8,
		momentum: float = 0.9,
		gradient_clipping: float = 1.0,
		pretrain: bool = False
):
	# 1. Create dataset
	obs_path = 'data/dataset' + str(dataset_id) + '/observe.tif'
	observe = np.array(Image.open(obs_path))
	observe = torch.from_numpy(observe).unsqueeze(0)
	dataset = ImgDataset(image, mask, observe, patch_size, stride, N)

	# 2. Split into train / validation partitions
	val_percent = 0.2
	n_val = int(len(dataset) * val_percent)
	n_train = len(dataset) - n_val
	train_set, valid_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

	# 3. Create data loaders
	loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
	train_loader = DataLoader(train_set, shuffle=True, **loader_args)
	valid_loader = DataLoader(valid_set, shuffle=False, drop_last=True, **loader_args)

	logging.info(f'''Starting training:
		Epochs:          {epochs}
		Batch size:      {batch_size}
		Learning rate:   {learning_rate}
		Device:          {device.type}
		Patch size:		 {patch_size}
	''')

	# 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
	optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
	grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
	criterion = nn.BCEWithLogitsLoss()
	global_step = 0
	early_stop = EarlyStopping(patience=10)

	# 5. Begin training
	pooling = nn.AvgPool2d(N, N)

	start = time.time()
	for epoch in range(1, epochs + 1):
		model.train()
		epoch_loss = 0

		num_train_batches = len(train_loader)
		acc = 0
		with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
			for batch in train_loader:
				images, true_masks, observed_masks = batch['image'], batch['mask'], batch['observe']
				assert images.shape[1] == model.n_channels, \
					f'Network has been defined with {model.n_channels} input channels, ' \
					f'but loaded images have {images.shape[1]} channels. Please check that ' \
					'the images are loaded correctly.'

				images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
				true_masks = true_masks.to(device=device, dtype=torch.float32)
				observed_masks = observed_masks.to(device=device, dtype=torch.float32)

				with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
					masks_pred = model(images)
					masks_pred = pooling(masks_pred)
					loss = criterion(masks_pred, true_masks.float())

					true_masks = (true_masks > 0.5).float()
					masks_pred = (masks_pred > 0.5).float()
					acc += torch.sum(masks_pred == true_masks) / torch.numel(masks_pred)

				if not pretrain:
					loss -= get_annotator_value(model.t_matrix, observed_masks, true_masks)

				optimizer.zero_grad(set_to_none=True)
				grad_scaler.scale(loss).backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
				grad_scaler.step(optimizer)
				grad_scaler.update()

				pbar.update(images.shape[0])
				global_step += 1
				epoch_loss += loss.item()
				pbar.set_postfix(**{'loss (batch)': loss.item()})

		epoch_loss = epoch_loss / num_train_batches
		scheduler.step(epoch_loss)
		logging.info('Current learning rate: {}'.format(scheduler._last_lr))
		logging.info('Training loss: {}'.format(epoch_loss))

		acc = acc / num_train_batches
		logging.info('Train Acc: {}'.format(acc))
		print('Train Acc: {}'.format(acc))

		# evaluate
		model.eval()
		epoch_loss = 0
		num_valid_batches = len(valid_loader)
		acc = 0
		for batch in valid_loader:
			images, true_masks, observed_masks = batch['image'], batch['mask'], batch['observe']

			images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
			true_masks = true_masks.to(device=device, dtype=torch.float32)
			observed_masks = observed_masks.to(device=device, dtype=torch.float32)

			with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
				masks_pred = model(images)
				masks_pred = pooling(masks_pred)
				loss = criterion(masks_pred, true_masks.float())

				true_masks = (true_masks > 0.5).float()
				masks_pred = (masks_pred > 0.5).float()
				acc += torch.sum(masks_pred == true_masks) / torch.numel(masks_pred)

			if not pretrain:
				loss -= get_annotator_value(model.t_matrix, observed_masks, true_masks)

			epoch_loss += loss.item()

		epoch_loss = epoch_loss / num_valid_batches
		logging.info('Validation loss: {}'.format(epoch_loss))

		acc = acc / num_valid_batches
		logging.info('Validation Acc: {}'.format(acc))
		print('Validation Acc: {}'.format(acc))

		early_stop(epoch_loss)
		if early_stop.early_stop:
			logging.info('Early stop!')
			print('Early stop!')
			break

	end = time.time()
	logging.info(f"Deep learning module training time: {(end - start):.3f} seconds")


def train_para(N, model, image, mask, dataset_id, device, pretrain=False, save_id=''):
	logging.info(f'----------------------Deep learning part START----------------------')
	print(f'----------------------Deep learning part START----------------------')
	logging.info('Now N = %d' % N)

	epochs = 100
	batch_size = 4
	lr = 1e-5
	patch_size = 100
	stride = 100

	model.to(device=device)
	train_model(model, device, image, mask, N, patch_size, stride, dataset_id, epochs, batch_size, lr, pretrain=pretrain)

	path = 'output/dataset' + str(dataset_id) + save_id + '/model_para'
	if not os.path.exists(path):
		os.mkdir(path)
	if not pretrain:
		torch.save(model, path + '/model_' + str(N) + '.pth')
	else:
		torch.save(model, path + '/pretrain.pth')

	logging.info(f'----------------------Deep learning part END----------------------')
	print(f'----------------------Deep learning part END----------------------')


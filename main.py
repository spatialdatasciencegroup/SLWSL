import logging
import os
import time
import numpy as np
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torch import nn

from train_para import train_para
from infer_label import inference
from unet import UNet


def parse_args():
	parser = argparse.ArgumentParser()

	#  Parameters
	parser.add_argument("--device", default='cuda', type=str)
	parser.add_argument("--dataset", default=1, type=int, help="Dataset ID")
	parser.add_argument("--eta", default=10, type=int, help="Resolution constant")
	parser.add_argument("-K", default=2, type=int, help="Resolution levels")
	parser.add_argument('--pretrain', action='store_true', help="Use pretrain parameters or not")


	return parser.parse_args()


def main(save_id):
	log_path = 'log/'
	if not os.path.exists(log_path):
		os.mkdir(log_path)
	log_path = log_path + str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))) + '.log'
	logging.basicConfig(filename=log_path,
						filemode='a',
						datefmt='%H:%M:%S',
						level=logging.INFO,
						format='%(asctime)s: %(message)s')

	args = parse_args()

	dataset_id = args.dataset
	eta = args.eta
	level_num = args.K

	batch_sizes = [32, 512, 4096]
	uncertain_thresholds = [0, 0.6, 0.5]

	image = Image.open('data/dataset' + str(dataset_id) + '/image.tif')
	image = transforms.ToTensor()(image)

	device = torch.device(args.device)
	logging.info(f'Train on device {device}')
	print(f'Train on device {device}')

	output_path = 'output/'
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	path = output_path + 'dataset' + str(dataset_id) + save_id
	if not os.path.exists(path):
		os.mkdir(path)

	last_infer_path = ''

	if args.pretrain:
		mask = np.array(Image.open('data/dataset' + str(dataset_id) + '/observe.tif'))
		mask = torch.from_numpy(mask).unsqueeze(0)

		n_channels = 3
		n_classes = 1
		model = UNet(n_channels=n_channels, n_classes=n_classes)
		train_para(100, model, image, mask, dataset_id, device, True, save_id)
	else:
		model_path = 'output/dataset' + str(dataset_id) + '/pretrain.pth'
		model = torch.load(model_path, map_location=device)

	t_matrix = nn.Embedding(4, 1).to(device)
	model.t_matrix = t_matrix

	for i in range(level_num, -1, -1):
		N = eta ** i
		logging.info('i = %d, N = %d' % (i, N))
		print('i = %d, N = %d' % (i, N))

		batch_size = batch_sizes[-i - 1]
		t = uncertain_thresholds[-i - 1]

		# label inference
		inference(eta, i, batch_size, dataset_id, model, image, device, last_infer_path, t, save_id)

		# parameter update
		mask_path = path + '/label_map/label_map_' + str(N) + '.tif'
		mask = np.array(Image.open(mask_path))
		mask = torch.from_numpy(mask).unsqueeze(0)
		train_para(N, model, image, mask, dataset_id, device, save_id)

		# path for next iteration
		last_infer_path = path + '/label_map/label_map_' + str(N) + '.tif'


if __name__ == '__main__':
	main('_0')

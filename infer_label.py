import os
from torch import nn
import torch.utils.data as data
import logging
import time

from utils.data_loading import FormulaDataset
from utils.utils import *
from utils.hmt import get_tree


class PslLoss(nn.Module):
	def __init__(self):
		super(PslLoss, self).__init__()
		self.relu = nn.ReLU()

	def forward(self, interpretation, formulas_l, formulas_r, weight, p=1):
		d_f = self.relu(interpretation(formulas_l) - interpretation(formulas_r)).to(weight.device)
		l_f = weight.unsqueeze(1) * torch.pow(d_f, p)
		return torch.mean(l_f)


def get_formulas(eta, i, image_shape, dataset_id, last_infer_path='', uncertain_threshold=0.65):
	formulas = []
	weight = []

	height = image_shape[0]
	width = image_shape[1]

	parents, neighbors = get_tree(height, width, eta, i, dataset_id, last_infer_path, uncertain_threshold)

	for i, line in enumerate(tqdm(parents, desc='Loading topology formulas')):
		for p in line[1:]:
			formulas.append([i, p])
			weight.append(0.7)

	for i, line in enumerate(tqdm(neighbors, desc='Loading spatial neighbor formulas')):
		for x in line[1:]:
			formulas.append([i, x])
			weight.append(0.3)

	logging.info('# Variables: %d' % len(parents))
	print('# Variables: %d' % len(parents))
	logging.info('# Formulas: %d' % len(formulas))
	print('# Formulas: %d' % len(formulas))

	return formulas, weight, len(parents)


def initialization(eta, i, image_shape, model, image, device, last_infer_path='', uncertain_threshold=0.65):
	N = eta ** i
	width = image_shape[1] // N

	# initialize interpretation
	pred_value, pred_label = output_img(model, image, 100, device)
	pred_value = torch.sigmoid(torch.tensor(pred_value)).unsqueeze(0)
	pooling = nn.AvgPool2d(N, N)

	if len(last_infer_path) != 0:
		pred_value = pooling(pred_value).squeeze(0).numpy()
		_, area2value, _ = uncertain_mapping(width, eta, last_infer_path, uncertain_threshold, pred_value,
											 desc='Initialize')
		para = torch.tensor(list(area2value.values())).unsqueeze(1)
	else:
		para = pooling(pred_value)
		para = para.flatten().unsqueeze(1)
	interpretation = nn.Embedding.from_pretrained(para, freeze=False)

	return interpretation, para


def train(interpretation, initial_weight, data_iter, num_epochs, loss, optimizer, scheduler, early_stop, device, p=1):
	interpretation = interpretation.to(device)
	initial_weight = initial_weight.to(device)

	bce_loss = nn.BCELoss()
	start = time.time()
	for epoch in range(num_epochs):
		l_sum = 0
		num_batch = len(data_iter)
		for batch in tqdm(data_iter, desc='epoch ' + str(epoch)):
			f_l, f_r, w = [d.to(device) for d in batch]
			l = loss(interpretation, f_l, f_r, w, p)
			bce_l = bce_loss(torch.clamp(interpretation(f_l), 0, 1), initial_weight[f_l])
			l += 1e-2 * bce_l
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			l_sum += l.cpu().item()
		l_epoch = l_sum / num_batch
		logging.info('epoch %d, loss %f' % (epoch + 1, l_epoch))
		print('epoch %d, loss %f' % (epoch + 1, l_epoch))
		early_stop(l_epoch)
		if early_stop.early_stop:
			logging.info('Early stop!')
			break
		scheduler.step(l_epoch)

	end = time.time()
	logging.info(f"Logic inference module training time: {(end - start):.3f} seconds")


def inference(eta, i, batch_size, dataset_id, model, image, device, last_infer_path='',
			  uncertain_threshold=0.65, save_id=''):
	logging.info(f'----------------------Logic inference part START----------------------')
	print(f'----------------------Logic inference part START----------------------')

	N = eta ** i
	logging.info('Now i = %d, N = %d' % (i, N))

	image_shape = image.shape[1], image.shape[2]
	height = image_shape[0] // N
	width = image_shape[1] // N

	path = 'output/dataset' + str(dataset_id) + save_id + '/label_map'
	if not os.path.exists(path):
		os.mkdir(path)

	# ground rules
	formulas, weight, atom_num = get_formulas(eta, i, image_shape, dataset_id, last_infer_path, uncertain_threshold)

	# initialize interpretation
	interpretation, initial_weight = initialization(eta, i, image_shape, model, image, device, last_infer_path,
													uncertain_threshold)

	epoch = 300
	learning_rate = 1 / (10 ** (i + 1))
	momentum = 0.99
	p = 1

	optimizer = torch.optim.SGD(interpretation.parameters(), learning_rate, momentum)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
	early_stop = EarlyStopping(min_delta=1e-5)

	l = PslLoss()

	dataset = FormulaDataset(formulas, weight)
	data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

	train(interpretation, initial_weight, data_iter, epoch, l, optimizer, scheduler, early_stop, device, p)

	# store the inferred label map
	if len(last_infer_path) != 0:
		target2pixels, _, _ = uncertain_mapping(width, eta, last_infer_path, uncertain_threshold, desc='output')
		pixel_number = height * width
		label_map = np.array([-1] * pixel_number, dtype=np.float32)
		values = interpretation.cpu().weight.data.numpy()
		for i, v in enumerate(list(target2pixels.values())):
			label_map[v] = values[i][0]
		label_map.resize(height, width)
	else:
		label_map = interpretation.cpu().weight.data.resize(height, width).numpy()

	max_v = np.max(label_map)
	min_v = np.min(label_map)
	normalize = (label_map - min_v) / (max_v - min_v)

	Image.fromarray(normalize).save(path + '/label_map_' + str(N) + '.tif')

	logging.info(f'----------------------Logic inference part END----------------------')
	print(f'----------------------Logic inference part END----------------------')

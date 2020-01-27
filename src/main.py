# import libraries
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from multiprocessing import cpu_count
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import argparse
import os
# helper functions/Classes for dataset preprocessing
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import dataset_utils.transform_and_augment as trans_aug
from dataset_utils.dataset import TrainImageDataset, TestImageDataset
# additional functions/Classes for logging
from model_utils.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from model_utils.test_callbacks import PredictionsSaverCallback
from model_utils import helpers
# additional functions/Classes for PyTorch Models
from model_utils import classifier
from model_utils import unet
import torch.optim as optim

# ignore warning
import warnings
warnings.filterwarnings("ignore")



def create_call_backs(args):
	# Training callbacks
	tb_viz_cb = TensorboardVisualizerCallback(os.path.join(args.project_dir, 'sum_logs/tb_viz'))
	tb_logs_cb = TensorboardLoggerCallback(os.path.join(args.project_dir, 'sum_logs/tb_logs'))
	model_saver_cb = ModelSaverCallback(os.path.join(args.project_dir,'sum_logs/tb_logs/model.pt'), verbose=True)
	origin_img_size = 224
	pred_saver_cb = PredictionsSaverCallback(os.path.join(args.project_dir, 'data/output/submit.csv.gz'),
												 origin_img_size, args.threshold)

	return tb_viz_cb, tb_logs_cb, model_saver_cb, pred_saver_cb

def create_train_val_test_dataloaders(args, threads, use_cuda):
	# create a train dataset
	train_coco = COCO(os.path.join(args.train_annotations_small_path))
	# create a val dataset
	val_coco = COCO(os.path.join(args.val_annotations_small_path))
	# 
	train_ds = TrainImageDataset(img_dir = args.train_image_directory , cocodataset = train_coco, y_data = None, 
								 input_img_resize = args.input_img_resize,output_img_resize = args.output_img_resize, X_transform=trans_aug.augment_img)

	train_loader = DataLoader(train_ds, args.batch_size,sampler=RandomSampler(train_ds), num_workers=threads,pin_memory=use_cuda)

	valid_ds = TrainImageDataset(img_dir = args.val_image_directory, cocodataset = val_coco, y_data = None, 
								 input_img_resize = args.input_img_resize, output_img_resize = args.output_img_resize, X_transform=trans_aug.augment_img)

	valid_loader = DataLoader(valid_ds, args.batch_size,
							  sampler=SequentialSampler(valid_ds),
							  num_workers=threads,
							  pin_memory=use_cuda)
	# test dataset

	
	test_ds = TestImageDataset(img_dir = args.test_image_directory, img_resize = args.input_img_resize)
	test_loader = DataLoader(test_ds, args.batch_size,
							 sampler=SequentialSampler(test_ds),
							 num_workers=threads,
							 pin_memory=use_cuda)


	return train_loader, valid_loader, test_loader

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--project_dir', type=str, default='/home/webwerks/patricia/my-projects/github/Segmentation', help='project directory')
	
	parser.add_argument('--train_image_directory', type=str, default = "data/train/images")
	parser.add_argument('--train_annotations_path', type=str, default = "data/train/annotation.json")
	parser.add_argument('--train_annotations_small_path', type=str, default = "data/train/annotation-small.json")
	parser.add_argument('--val_image_directory', type=str, default = "data/val/images")
	parser.add_argument('--val_annotations_path', type=str, default = "data/val/annotation.json")
	parser.add_argument('--val_annotations_small_path', type=str, default = "data/val/annotation-small.json")
	parser.add_argument('--test_image_directory', type=str, default = "data/test")
	
	parser.add_argument('--input_img_resize', type=tuple, default = (224, 224), help='The resize size of the input images of the neural net')
	parser.add_argument('--output_img_resize', type=tuple, default = (224, 224), help='The resize size of the output images of the neural net')
	parser.add_argument('--batch_size', type=int, default = 10)
	parser.add_argument('--epochs', type=int, default = 50)
	parser.add_argument('--threshold', type=float, default = 0.5)
	parser.add_argument('--validation_size', type=float, default = 0.2)\
	# Put 'None' to work on full dataset or a value between 0 and 1
	parser.add_argument('--sample_size', type=float, default = None)

	# finetuning and inference arguments
	parser.add_argument('--output_model_file', type=str, default = None)
	parser.add_argument('--preliminary_training', type=bool, default = True)
	parser.add_argument('--finetuning', type=bool, default = True)
	parser.add_argument('--fine_tune_epochs', type=int, default = 10)
	parser.add_argument('--pred_on_inference_set', type=bool, default = False)
	parser.add_argument('--pred_on_inference_img', type=bool, default = False)

	args = parser.parse_args()
	os.chdir(args.project_dir)
	# -- Optional parameters
	threads = cpu_count()
	use_cuda = torch.cuda.is_available()

	train_loader, valid_loader, test_loader = create_train_val_test_dataloaders(args, threads, use_cuda)
	tb_viz_cb, tb_logs_cb, model_saver_cb, pred_saver_cb = create_call_backs(args)

	# train model

	net = unet.UNet16()
	if args.preliminary_training:
		net = unet.freezing_pretrained_layers(model = net, grad = False)
		unet_classifier = classifier.UnetClassifier(net, args.epochs)
		# Train the classifier
		unet_classifier.train(train_loader, valid_loader, args.epochs, callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])
	

	if args.finetuning:
		print("finetuning")
		net.load_state_dict(torch.load(os.path.join(args.project_dir,'sum_logs/tb_logs/model.pt')))
		net = unet.freezing_pretrained_layers(model = net)
		unet_classifier = classifier.UnetClassifier(net, args.fine_tune_epochs)
		# Train the classifier
		unet_classifier.train(train_loader, valid_loader, args.fine_tune_epochs, finetune = args.finetuning, callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])
	
	# if args.pred_on_inference_set:
	# 	origin_img_size = 224
	# 	# # Predict & save
	# 	classifier.predict(test_loader, callbacks=[pred_saver_cb])
	# 	pred_saver_cb.close_saver()

	# if args.pred_on_inference_img:
	# 	net.load_state_dict(torch.load(os.path.join(args.project_dir,'sum_logs/tb_logs/model.pt')))


if __name__ == '__main__':
	main()

import argparse
import sys
import scipy
import os
from PIL import Image
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from skimage import io, transform
from model import ModelFactory
from torch.autograd import Variable
import time
description='Video Super Resolution pytorch implementation'

def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='TDAN',
                    help='network architecture. Default False')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4, 
                    help='VSR scale. Default 4')
parser.add_argument('-t', '--test-set', metavar='NAME', type=str, default='data/test_vsr',
                    help='dataset for testing.')
parser.add_argument('-mp', '--model-path', metavar='MP', type=str, default='model',
                    help='dataset for testing. Default IndMya')
parser.add_argument('-sp', '--save-path', metavar='SP', type=str, default='res',
                    help='saving')
args = parser.parse_args()

model_factory = ModelFactory()
model = model_factory.create_model(args.model)
dir_data = args.test_set
dir_LR = args.test_set#os.path.join(dir_data, "LQ")
lis = sorted(os.listdir(dir_LR))
model_path = os.path.join(args.model_path, 'model.pt')
if not os.path.exists(model_path):
	raise Exception('Cannot find %s.' %model_path)
model = torch.load(model_path)
model.eval()
path = args.save_path
if not os.path.exists(path):
            os.makedirs(path)
t = 0
tc = 0
for i in range(len(lis)):
	print(lis[i])
	LR = os.path.join(dir_LR, lis[i])
	ims = sorted(os.listdir(LR))
	num = len(ims)
	# number of the seq
	num = len(ims)

        # get frame size
	image = io.imread(os.path.join(LR, ims[0]))
	row, col, ch = image.shape
	frames_lr = np.zeros((5, int(row), int(col), ch))
	for j in range(num):
		#if (j + 1)%10 != 0:
		#	continue 
		# for boundary frames
		for k in range(j-2, j + 3):
			idx = k-j+2
			if k < 0:
				k = -k
			if k >= num:
				k = num - 3
			
			frames_lr[idx, :, :, :] = io.imread(os.path.join(LR, ims[k]))
		start = time.time()
		frames_lr = frames_lr/255.0 - 0.5
		lr = torch.from_numpy(frames_lr).float().permute(0, 3, 1, 2)
		lr = Variable(lr.cuda()).unsqueeze(0).contiguous()
		output, _ = model(lr)
		#output = forward_x8(lr, model)
		output = (output.data + 0.5)*255
		output = quantize(output, 255)
		output = output.squeeze(dim=0)
		elapsed_time = time.time() - start
		t += elapsed_time
		tc += 1
		print(elapsed_time)
			
		img_name = os.path.join(path, lis[i] + '_' + ims[j])
		Image.fromarray(np.around(output.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)).save(img_name)
print('avg running time:', t/tc)
        

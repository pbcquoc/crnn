import torch
from torch.autograd import Variable
from models.utils import strLabelConverter, resizePadding
from PIL import Image
import sys
import models.crnn as crnn
import argparse
from torch.nn.functional import softmax
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True, help='path to img')
parser.add_argument('--alphabet', required=True, help='path to vocab')
parser.add_argument('--model', required=True, help='path to model')
parser.add_argument('--imgW', type=int, default=None, help='path to model')
parser.add_argument('--imgH', type=int, default=32, help='path to model')

opt = parser.parse_args()


alphabet = open(opt.alphabet).read().rstrip()
nclass = len(alphabet) + 1
nc = 3

model = crnn.CRNN(opt.imgH, nc, nclass, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % opt.model)
model.load_state_dict(torch.load(opt.model, map_location='cpu'))

converter = strLabelConverter(alphabet, ignore_case=False)

image = Image.open(opt.img).convert('RGB')
image = resizePadding(image, opt.imgW, opt.imgH)

if torch.cuda.is_available():
    image = image.cuda()

image = image.view(1, *image.size())
image = Variable(image)

model.eval()

start_time = time.time()
preds = model(image)

values, prob = softmax(preds, dim=-1).max(2)
preds_idx = (prob > 0).nonzero()
sent_prob = values[preds_idx[:,0], preds_idx[:, 1]].mean().item()

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)
preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s : prob: %s time: %s' % (raw_pred, sim_pred, sent_prob, time.time() - start_time))



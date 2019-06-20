import torch
from torch.autograd import Variable
import utils
from PIL import Image
import sys
import models.crnn as crnn
import argparse
from torch.nn.functional import softmax
import numpy as np
from loader import resizeNormalize
parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True, help='path to img')
parser.add_argument('--alphabet', required=True, help='path to vocab')
parser.add_argument('--model', required=True, help='path to model')
parser.add_argument('--imgW', type=int, default=1000, help='path to model')
parser.add_argument('--imgH', type=int, default=64, help='path to model')

opt = parser.parse_args()


alphabet = open(opt.alphabet).read().rstrip()
nclass = len(alphabet) + 1

model = crnn.CRNN(32, 1, nclass, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % opt.model)
model.load_state_dict(torch.load(opt.model))

converter = utils.strLabelConverter(alphabet)

transformer = resizeNormalize(opt.imgW, opt.imgH)
image = Image.open(opt.img).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

#preds_ = Variable(torch.FloatTensor(preds.detach().cpu().numpy().astype('float')))
#preds_ = beam_decoder.beamsearch(softmax(preds_, -1))
#print(beam_decoder.decode(preds_))
values, prob = softmax(preds, dim=-1).max(2)
preds_idx = (prob > 0).nonzero()
sent_prob = values[preds_idx[:,0], preds_idx[:, 1]].mean().item()

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)
preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s : prob: %s' % (raw_pred, sim_pred, sent_prob))



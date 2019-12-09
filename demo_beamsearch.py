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
parser.add_argument('--imgW', type=int, default=1024, help='path to model')
parser.add_argument('--imgH', type=int, default=64, help='path to model')

parser.add_argument('--beamsearch_lib', type=str, required=True, help='path to beamsearch lib')
parser.add_argument('--corpus', type=str, required=True, help='path to corpus to learn language model')
parser.add_argument('--word_chars', type=str, required=True, help='path to word chars which was removed number')

opt = parser.parse_args()


alphabet = open(opt.alphabet).read().rstrip()
nclass = len(alphabet) + 1

model = crnn.CRNN(32, 1, nclass, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % opt.model)
model.load_state_dict(torch.load(opt.model, map_location='cpu'))

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

beam_decoder = utils.BeamSearchDecoder(
    lib=opt.beamsearch_lib,
    corpus=opt.corpus,
    chars=opt.alphabet,
    word_chars=opt.word_chars)

preds_ = Variable(torch.FloatTensor(preds.detach().cpu().numpy().astype('float')))
preds_ = beam_decoder.beamsearch(softmax(preds_, -1))
print(beam_decoder.decode(preds_))



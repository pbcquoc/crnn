from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import models.utils as utils
from loader import DatasetLoader
from multiprocessing import cpu_count
from tqdm import tqdm
from torchsummary import summary

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, help='path to root folder')
parser.add_argument('--train', required=True, help='path to dataset')
parser.add_argument('--val', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=512, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, required=True, help='path to char in labels')
parser.add_argument('--expr_dir', required=True, type=str, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


loader = DatasetLoader(opt.root, opt.train, opt.val, opt.imgW, opt.imgH)
train_loader = loader.train_loader(opt.batch_size, num_workers=opt.workers) 
test_loader = loader.test_loader(opt.batch_size, num_workers=opt.workers)

alphabet = open(os.path.join(opt.root, opt.alphabet)).read().rstrip()
nclass = len(alphabet) + 1
nc = 3

print(len(alphabet), alphabet)
converter = utils.strLabelConverter(alphabet, ignore_case=False)
criterion = CTCLoss()

crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    pretrain = torch.load(opt.pretrained)
    crnn.load_state_dict(pretrain, strict=False)

image = torch.FloatTensor(opt.batch_size, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batch_size * 5)
length = torch.IntTensor(opt.batch_size)

if opt.cuda:
    crnn.cuda()
    image = image.cuda()
    criterion = criterion.cuda()

summary(crnn.cnn, (3, opt.imgH, opt.imgW))

image = Variable(image)
text = Variable(text)
length = Variable(length)
train_loss_avg = utils.averager()
train_cer_avg = utils.averager()

# setup optimizer
optimizer = optim.Adam(crnn.parameters(), lr=opt.lr)

def val(net, data_loader, criterion, max_iter=1000):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    
    val_iter = iter(data_loader)

    val_loss_avg = utils.averager()
    val_cer_avg = utils.averager()
    max_iter = min(max_iter, len(data_loader))
    with torch.no_grad():
        for i in range(max_iter):
            data = val_iter.next()
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)

            preds = crnn(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length)/batch_size
            cost = cost.detach().item()
            val_loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            cer_loss = utils.cer_loss(sim_preds, cpu_texts)
            val_cer_avg.add(cer_loss)


    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-30s => %-30s, gt: %-30s' % (raw_pred, pred, gt))

    print('Test loss: %f - cer loss %f' % (val_loss_avg.val(), val_cer_avg.val()))


def trainBatch(net, data, criterion, optimizer):
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length)/batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    cost = cost.detach().item()

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    cer_loss = utils.cer_loss(sim_preds, cpu_texts)
    return cost, cer_loss, len(cpu_images)


for epoch in range(1, opt.nepoch+1):
    t = tqdm(iter(train_loader), total=len(train_loader), desc='Epoch {}'.format(epoch))
    for i, data in enumerate(t):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost, cer_loss, n = trainBatch(crnn, data, criterion, optimizer)       

        train_loss_avg.add(cost)
        train_cer_avg.add(cer_loss)

    print('[%d/%d] Loss: %f - cer loss: %f' %
            (epoch, opt.nepoch, train_loss_avg.val(), train_cer_avg.val()))
    train_loss_avg.reset()
    train_cer_avg.reset()

    if epoch % opt.valInterval == 0:
        val(crnn, test_loader, criterion)

    # do checkpointing
    if epoch % opt.saveInterval == 0:
        torch.save(
            crnn.state_dict(), '{}/netCRNN_{}.pth'.format(opt.expr_dir, epoch))

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
import utils
import dataset
from loader import DatasetLoader
from multiprocessing import cpu_count

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, help='path to root folder')
parser.add_argument('--train', required=True, help='path to dataset')
parser.add_argument('--val', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=1024, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, required=True, help='path to char in labels')
parser.add_argument('--save_model', default='expr/net.pth', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=5, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=5, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
opt = parser.parse_args()
print(opt)

expr_dir, _ = os.path.split(opt.save_model)
if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


loader = DatasetLoader(opt.root, opt.train, opt.val, opt.imgW, opt.imgH)
train_loader = loader.train_loader(opt.batch_size, num_workers=cpu_count()) 
test_loader = loader.test_loader(opt.batch_size, num_workers=cpu_count())

alphabet = open(opt.alphabet).read().rstrip()
nclass = len(alphabet) + 1
nc = 1

print(len(alphabet), alphabet)
converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    pretrain = torch.load(opt.pretrained)
    crnn.load_state_dict(pretrain, strict=False)

image = torch.FloatTensor(opt.batch_size, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batch_size * 5)
length = torch.IntTensor(opt.batch_size)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)


# setup optimizer
optimizer = optim.Adam(crnn.parameters(), lr=opt.lr)

def val(net, data_loader, criterion, max_iter=1000):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    
    val_iter = iter(data_loader)

    i = 0
    loss_avg = 0
    cer_loss_avg = 0
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length)
        loss_avg += cost

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cer_loss = utils.cer_loss(sim_preds, cpu_texts, ignore_case=False)
        cer_loss_avg += cer_loss

    loss_avg /= len(data_loader.dataset)
    cer_loss_avg /= len(data_loader.dataset)

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-30s => %-30s, gt: %-30s' % (raw_pred, pred, gt))

    print('Test loss: %f - cer loss %f' % (loss_avg, cer_loss_avg))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) 
    crnn.zero_grad()
    cost.backward()
    optimizer.step()

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    cer_loss = utils.cer_loss(sim_preds, cpu_texts, ignore_case=False)

    return cost, cer_loss


for epoch in range(1, opt.nepoch+1):
    train_iter = iter(train_loader)
    loss_avg = 0
    cer_avg = 0
    i = 0

    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost, cer_loss = trainBatch(crnn, criterion, optimizer)        
        loss_avg += cost
        cer_avg += cer_loss
        i += 1
    
    loss_avg/=len(train_loader.dataset)
    cer_avg/=len(train_loader.dataset)

    if epoch % opt.displayInterval == 0:
        print('[%d/%d] Loss: %f - cer loss: %f' %
                (epoch, opt.nepoch, loss_avg, cer_avg))

    if epoch % opt.valInterval == 0:
        val(crnn, test_loader, criterion)

    # do checkpointing
    if epoch% opt.saveInterval == 0:
        torch.save(
            crnn.module.state_dict(), '{}.epoch_{}'.format(opt.save_model, epoch))

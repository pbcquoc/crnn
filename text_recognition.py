import argparse
import os
from PIL import Image
import numpy as np
import cv2
import models.crnn as crnn
import models.utils as utils
import torch
from torch.nn.functional import softmax
from torch.autograd import Variable
import time
import io
from google.cloud import vision
import base64
from google.cloud.vision import types
import pytesseract

class TextRecognition:
    def __init__(self, weights, char, height=32, cuda=None):
        alphabet = open(char).read().rstrip()
        nclass = len(alphabet) + 1
        self.height = height
        self.device = 'cuda:{}'.format(cuda) if cuda != None else 'cpu'
        self.model = crnn.CRNN(32, 3, nclass, 256)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        if cuda != None:
            self.model.cuda(self.device)
        self.converter = utils.strLabelConverter(alphabet, ignore_case=False) 
        self.model.eval()

        self.client = vision.ImageAnnotatorClient()

    def predict(self, image):
        image = Image.fromarray(image)
        image = utils.resizePadding(image, None, self.height)
        image = image.view(1, *image.size())
        image = Variable(image)

        image = image.to(self.device)

        preds = self.model(image)
        preds = preds.squeeze(1)
        sim_pred, sent_prob = self.decode(preds)

        return sim_pred, sent_prob

    def decode(self, preds):
        values, prob = softmax(preds, dim=-1).max(-1)
        preds_idx = (prob > 0).nonzero().squeeze(-1)

        sent_prob = values[preds_idx].mean().item()

        _, preds = preds.max(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))

        preds = preds.view(-1)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        
        return sim_pred, sent_prob

    def predicts(self, images):
        images = [Image.fromarray(image) for image in images]
        sizes = [image.size for image in images]
        maxW = utils.maxWidth(sizes, self.height)
        images = [utils.resizePadding(image, maxW, self.height) for image in images]
        image = torch.cat([t.unsqueeze(0) for t in images], 0)
        
        image = image.to(self.device)
        preds = self.model(image)
    
        rs = []
        for i in range(len(images)):
            sim_pred, sent_prob = self.decode(preds[:, i, :])
            rs.append((sim_pred, sent_prob))
        return rs

    def binary(self, img_gray):
        _, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        return threshold

    def detect(self, rois):
        rs = []
        for roi in rois:
            img = np.array(Image.fromarray(roi['img']))
#            img = self.binary(img)
#            img = np.array(Image.fromarray(img).convert('RGB'))
#            img_bin = Image.fromarray(img)
#            img_bin.save('/data/quocpbc/tmp/field_bin_{}.jpg'.format(roi['name']))
            text, prob = self.google_vision([img])
            r = {'name': roi['name'], 'prob':roi['prob']*prob, 'text': text}
            rs.append(r)
        return rs
    
    def google_vision(self, img):
        img = img[0]
        img = Image.fromarray(img)
        img.save('tmp.jpg')
        with io.open('tmp.jpg', 'rb') as image_file:
            content = image_file.read()

        content = types.Image(content=content)

        response = self.client.text_detection(image=content)
        texts = response.text_annotations
        if len(texts) > 0:
            return texts[0].description.rstrip(), 1.0
        else:
            return "", 0

    def tesseract(self, img):
        img = img[0]
        img = Image.fromarray(img)
        target = pytesseract.image_to_string(img, lang='eng', config='--psm 8 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return target, 1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, type=str, help='path to img')
    parser.add_argument('--char', required=True, type=str, help='path to dictionary')
    parser.add_argument('--weights', required=True, type=str, help='path to pretrained model')
    parser.add_argument('--gpu', type=int, default=None, help='cuda device')

    args = parser.parse_args()
    print (args)
    
    detector = TextRecognition(weights=args.weights, char=args.char, cuda=args.gpu)
    image = Image.open(args.img).convert('RGB')
    images = [np.array(image), np.array(image)]
    start_time = time.time()

    r  = detector.predicts(images)
    print('elasped time: ', time.time() - start_time)
    print(r)

if __name__ == '__main__':
    main()

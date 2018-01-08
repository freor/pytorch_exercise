import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import time, os, json
import numpy as np

from coco_utils import *
from captioning import *

from option import *
#from evals import *

model_path='./models'
data_path ='./coco_captioning'


# Load Data
train_data = load_coco_data(base_dir=data_path, max_train=50000)

captions = train_data['train_captions']
img_idx  = train_data['train_image_idxs']
img_features = train_data['features'][img_idx]
word_to_idx = train_data['word_to_idx']
idx_to_word = train_data['idx_to_word']

n_words = len(word_to_idx)
maxlen = train_data['train_captions'].shape[1]
input_dimension = train_data['features'].shape[1]

print("n_words", n_words)
print("maxlen", maxlen)
print("img_features", img_features.shape)
print("captions", captions.shape)
print(captions)

vcaptions = train_data['val_captions']
vimg_idx  = train_data['val_image_idxs']
vimg_features = train_data['features'][vimg_idx]
print("vimg_features", vimg_features.shape)
print("vcaptions", vcaptions.shape)


def evaluate_model(data, split):
    BLEUscores = {}

    minibatch = sample_coco_minibatch(data, split=split, batch_size="All")
    gt_captions, features, urls = minibatch # features: (10000, 512)
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    pr_captions = image_captioning(features)
    pr_captions = decode_captions(pr_captions, data['idx_to_word'])

    total_score = 0.0

    for gt_caption, pr_caption, url in zip(gt_captions, pr_captions, urls):
        total_score += BLEU_score(gt_caption, pr_caption)

    BLEUscores[split] = total_score / len(pr_captions)

    for split in BLEUscores:
        print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))


def image_captioning(features):
    pr_captions = np.zeros((features.shape[0], maxlen), int)

    captioning = torch.load("./myModel.pth")

    pr_captions = captioning.prediction(features, word_to_idx, idx_to_word)

    return pr_captions


captioning = Captioning()
captioning.cuda()

criterion = nn.CrossEntropyLoss() # = log softmax + nll loss
optimizer = torch.optim.Adam(captioning.parameters(), lr=1e-4, eps=1e-3, weight_decay=1e-4) # L2?

for e in range(epoch_size):

    total_loss = 0
    for b in range(captions.shape[0] // batch_size):
        imgs = Variable(torch.cuda.FloatTensor(img_features[b * batch_size: (b+1) * batch_size])).view(1, batch_size, -1) # (1, 512, 512)

        input_cap = Variable(torch.cuda.LongTensor(captions[b * batch_size: (b+1) * batch_size]))
        target_caps = input_cap.view(batch_size * 17)
        #target_cap = Variable(torch.cuda.LongTensor(captions[b * batch_size: (b+1) * batch_size, 1:])).view(512 * 16)

        out = captioning((imgs, captions[b * batch_size: (b+1) * batch_size])) #input_cap))
        prediction = out

        prediction = prediction.view(batch_size * 17, -1)
        loss = criterion(prediction, target_caps)
        total_loss += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if e % 10 == 1:
        print("epoch: %d, total_loss: %f" % (e, total_loss))
        torch.save(captioning, './myModel.pth')
        evaluate_model(train_data, 'val')



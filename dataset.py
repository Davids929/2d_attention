#coding=utf-8
import os
import cv2
import math
import json
import random
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms

PAD = 0
SOS = 1
EOS = 2
UNK = 3
random.seed(123)

train_transform_fn = transforms.Compose([
    transforms.RandomBrightness(0.3),
    transforms.RandomContrast(0.3),
    transforms.RandomSaturation(0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], 
                         [0.2023, 0.1994, 0.2010])
])

test_transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], 
                         [0.2023, 0.1994, 0.2010])
])

class FixedSizeDataset(Dataset):
    def __init__(self, line_path, voc_path, short_side=32, fix_width=128, 
                max_len=60, use_augment=False, add_symbol=True, max_sample_num=10000):

        self.short_side  = short_side
        self.fix_width   = fix_width
        self.max_len     = max_len
        self.use_augment = use_augment
        self.add_symbol  = add_symbol
        self.max_sample_num = max_sample_num
        self.id2word, self.word2id = self._load_voc_dict(voc_path, add_symbol=add_symbol)
        self.word_list = list(self.word2id.keys())
        self.imgs_list, self.labs_list = self._load_imgs(line_path)

    def _load_voc_dict(self, dict_path, add_symbol=True):
        id2word_dict = {}
        word2id_dict = {}
        if add_symbol:
            id2word_dict = {PAD:'<pad>', SOS:'<s>', EOS:'</s>', UNK:'<unk>'}
            word2id_dict = {'<pad>':PAD, '<s>':SOS, '</s>':EOS, '<unk>':UNK}
        with open(dict_path, 'r', encoding='utf-8') as fi:
            line_list = fi.readlines()
        for idx, line in enumerate(line_list):
            if add_symbol:
                idx = idx + 4
            word = line.strip()[0]
            id2word_dict[idx]  = word
            word2id_dict[word] = idx
        return id2word_dict, word2id_dict

    def _load_imgs(self, line_path_list):
        imgs_list = []
        labs_list = []
        if not isinstance(line_path_list, list):
            line_path_list = [line_path_list]
        for line_path in line_path_list:
            with open(line_path, 'r', encoding='utf-8') as fi:
                #line_list = fi.readlines()
                for i, line in enumerate(fi):
                    if self.max_sample_num is not None:
                        if i>=self.max_sample_num:
                            break
                    lst = line.strip().split('\t')
                    if len(lst) == 1:
                        continue
                    img_path = lst[0]
                    label    = lst[1]
                    if not os.path.exists(img_path):
                        continue
                    imgs_list.append(img_path)
                    labs_list.append(label)
        return imgs_list, labs_list

    def __len__(self):
        return len(self.imgs_list)

    def text2ids(self, text, text_len, add_end_symbol=True):
        ids       = mx.nd.ones(shape=(text_len), dtype='float32')*EOS
        ids_mask  = mx.nd.zeros(shape=(text_len), dtype='float32')
        char_list = list(text)
        if add_end_symbol:
            char_list = char_list + ['</s>']
        for i, ch in enumerate(char_list):
            if ' ' == ch:
                continue
            if ch in self.word_list: 
                ids[i] = self.word2id[ch]  
            else:
                ids[i] = UNK
            ids_mask[i] = 1.0
        return ids, ids_mask

    def ids2text(self, ids, use_end_symbol=True):
        if not isinstance(ids, list):
            ids = list(ids)
        text_list = []
        for i in ids:
            int_i = int(i)
            if int_i == EOS and use_end_symbol:
                break
            text_list.append(self.id2word[int_i])
        return text_list

    def image_resize(self, img_np, max_width=512):
        h, w = img_np.shape[:2]
        if h > w:
            img_np = np.rot90(img_np)
            h, w = w, h
        if self.fix_width is not None:
            img_np = cv2.resize(img_np, (self.fix_width, self.short_side))
            return img_np
        w = int(w*self.short_side/h)
        if w> max_width:
            w = max_width
        img_np = cv2.resize(img_np, (w, self.short_side))
        return img_np

    def __getitem__(self, idx):
        img_path = self.imgs_list[idx]
        text     = self.labs_list[idx]
        img_np   = cv2.imread(img_path)
        img_np   = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np   = self.image_resize(img_np)
        img_nd   = mx.nd.array(img_np)
        h, w = img_nd.shape[:2]
        if self.use_augment:
            img_nd = train_transform_fn(img_nd)
        else:
            img_nd = test_transform_fn(img_nd)
        img_mask = mx.nd.ones(shape=(1, h, w), dtype='float32')
        lab, lab_mask = self.text2ids(text, self.max_len, add_end_symbol=self.add_symbol)

        return img_nd, img_mask, lab, lab_mask

class BucketDataset(FixedSizeDataset):
    def __init__(self, line_path, voc_path, short_side=32, fix_width=None,
                 max_len=60, use_augment=False, add_symbol=True, max_sample_num=10000,
                 load_bucket_path=None):
        super(BucketDataset, self).__init__(line_path, voc_path, 
                                            short_side=short_side,
                                            fix_width=fix_width,
                                            max_len=max_len,
                                            use_augment=use_augment,
                                            add_symbol=add_symbol,
                                            max_sample_num=max_sample_num)
        self.fix_width       = None
        self.split_width_len = self.short_side*4
        self.max_width       = short_side*16
        self.bucket_path     = load_bucket_path

        if self.bucket_path:
            if os.path.exists(self.bucket_path):
                self.load_bucket(self.bucket_path)
            else:
                self.gen_bucket(save_bucket=True)
        else:
            self.gen_bucket(save_bucket=False)

    def _get_bucket_key(self, img_shape):
        h, w = img_shape[:2]
        if h > w:
            h, w = w, h
        if w/h > self.max_width/self.split_width_len:
            return (self.short_side, self.max_width, self.max_len)
        ratio = math.ceil(self.short_side * w / h / self.split_width_len)

        return (self.short_side, self.split_width_len * ratio, self.max_len)

    def load_bucket(self, bucket_path):
        with open(bucket_path, 'r') as fi:
            self.bucket_dict = json.load(fi)
        self.bucket_keys = list(self.bucket_dict.keys())

    def gen_bucket(self, save_bucket=True):
        bucket_keys, bucket_dict = [], {}
        for idx in range(len(self.imgs_list)):
            img_np = cv2.imread(self.imgs_list[idx])
            text = self.labs_list[idx]
            if img_np is None:
                continue
            if len(text) >= self.max_len:
                continue
            bucket_key = self._get_bucket_key(img_np.shape)
            bucket_key = str(bucket_key)
            if bucket_key not in bucket_keys:
                bucket_keys.append(bucket_key)
                bucket_dict[bucket_key] = []
            bucket_dict[bucket_key].append(idx)

        for key in bucket_keys:
            print('bucket key:', key, 'the number of image:', len(bucket_dict[key]))

        self.bucket_dict = bucket_dict
        self.bucket_keys = bucket_keys
        if save_bucket:
            with open(self.bucket_path, 'w') as fi:
                json.dump(bucket_dict, fi)

    def __getitem__(self, idx):
        img_path = self.imgs_list[idx]
        text = self.labs_list[idx]
        img_np = cv2.imread(img_path)
        inp_h, inp_w, max_len = self._get_bucket_key(img_np.shape)
        source_data = mx.nd.zeros(shape=(3, inp_h, inp_w), dtype='float32')
        source_mask = mx.nd.zeros(shape=(1, inp_h, inp_w), dtype='float32')
        #img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np = self.image_resize(img_np, max_width=self.max_width)
        img_nd = mx.nd.array(img_np)
        h, w = img_nd.shape[:2]
        if self.use_augment:
            img_nd = train_transform_fn(img_nd)
        else:
            img_nd = test_transform_fn(img_nd)
        source_data[:, :h, :w] = img_nd
        source_mask[:, :h, :w] = 1.0
        lab, lab_mask = self.text2ids(text, max_len, add_end_symbol=self.add_symbol)
        return source_data, source_mask, lab, lab_mask, idx

class Sampler(object):
    def __init__(self, idx_list):
        self.idx_list = idx_list

    def __iter__(self):
        return iter(self.idx_list)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        return self.idx_list[item]

class BucketSampler(object):
    '''
    last_batch : {'keep', 'discard'}
        Specifies how the last batch is handled if batch_size does not evenly
        divide sequence length.

        If 'keep', the last batch will be returned directly, but will contain
        less element than `batch_size` requires.

        If 'discard', the last batch will be discarded.

    '''
    def __init__(self, batch_size, bucket_dict, shuffle=True, last_batch='discard'):
        bucket_keys  = list(bucket_dict.keys())
        self._batch_size = batch_size
        self._last_batch = last_batch
        self.shuffle     = shuffle
        self.sampler_list = []
        for key in bucket_keys:
            self.sampler_list.append(Sampler(bucket_dict[key]))

    def __iter__(self):
        if self.shuffle:
            for sampler in self.sampler_list:
                random.shuffle(sampler.idx_list)
            random.shuffle(self.sampler_list)

        # for _sampler in self.sampler_list:
        #     batch = []
        #     if self.shuffle:
        #         random.shuffle(_sampler.idx_list)
        #     for i in _sampler:
        #         batch.append(i)
        #         if len(batch) == self._batch_size:
        #             yield batch
        #             batch = []
        #     if batch:
        #         if self._last_batch == 'keep':
        #             yield batch
        #         elif self._last_batch == 'discard':
        #             continue
        #         else:
        #             raise ValueError(
        #                 "last_batch must be one of 'keep', 'discard', or 'rollover', " \
        #                 "but got %s"%self._last_batch)
        num_sampler = len(self.sampler_list)
        sampler_idx_list = list(range(num_sampler))
        start_idx_list = [0] * num_sampler
        while True:
            if sampler_idx_list == []:
                break
            samp_idx = random.sample(sampler_idx_list, 1)[0]
            _sampler = self.sampler_list[samp_idx]
            start_idx = start_idx_list[samp_idx]
            batch = []
            while True:
                if len(batch) == self._batch_size:
                    start_idx_list[samp_idx] = start_idx
                    break

                if start_idx < len(_sampler):
                    batch.append(_sampler[start_idx])
                    start_idx = start_idx + 1
                else:
                    sampler_idx_list.remove(samp_idx)
                    if self._last_batch == 'discard':
                        batch = []
                    break
            if batch:
                yield batch


    def __len__(self):
        num = 0
        for _sampler in self.sampler_list:
            if self._last_batch == 'keep':
                #num += (len(_sampler) + self._batch_size - 1) // self._batch_size
                num += math.ceil(len(_sampler/self._batch_size))
            elif self._last_batch == 'discard':
                num += len(_sampler) // self._batch_size
            else:
                raise ValueError(
                    "last_batch must be one of 'keep', 'discard', or 'rollover', " \
                    "but got %s" % self._last_batch)
        return num

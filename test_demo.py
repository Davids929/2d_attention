#coding=utf-8
import cv2
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
import numpy as np

PAD, SOS, EOS, UNK = 0, 1, 2, 3
json_path   = './data/attention_model-symbol.json'
params_path = './data/attention_model-0000.params'
voc_path    = './data/char_std_5990.txt'
short_side, max_width = 32, 512

transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], 
                         [0.2023, 0.1994, 0.2010])
])

class Tester(object):
    def __init__(self, gpu=0):

        self.id2word, _ = self._load_voc_dict(voc_path)
        self.ctx   = mx.gpu(gpu)
        self.model = gluon.SymbolBlock.imports(json_path, ['data0', 'data1'], 
                                               params_path, ctx=self.ctx) 
        self.batch_size = 16

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

    def ids2text(self, ids, use_end_symbol=True):
        if not isinstance(ids, list):
            ids = list(ids)
        text_list = []
        for i in ids:
            int_i = int(i)
            if int_i == EOS and use_end_symbol:
                break
            if int_i == UNK:
                continue
            text_list.append(self.id2word[int_i])
        text = ''.join(text_list)
        return text

    def image_resize(self, img_np):
        h, w = img_np.shape[:2]
        if h > w:
            img_np = np.rot90(img_np)
            h, w = w, h
        w = int(w*short_side/h)
        if w> max_width:
            w = max_width
        img_np = cv2.resize(img_np, (w, short_side))
        return img_np
    
    def single_test(self, image_path):
        img_np = cv2.imread(image_path)
        img_np = self.image_resize(img_np)
        img_np = cv2.cvt_color(img_np, cv2.COLOR_BRG2RGB)
        h, w   = img_np.shape[:2]
        img_nd = mx.nd.array(img_np)
        img_nd = transform_fn(img_nd).expand_dims(axis=0).as_in_context(self.ctx)
        mask   = mx.nd.ones((1, 1, h, w), dtype='float32').as_in_context(self.ctx)
        _, pred2, atten_mask = self.model(img_nd, mask)
        predict = mx.nd.softmax(pred2).asnumpy()
        predict = np.argmax(predict, axis=-1)
        text    = self.ids2text(predict[0])
        return text

    def batch_test(self, batch_imgs):
        batch_size = len(batch_imgs) 
        data = mx.nd.zeros((batch_size, 3, short_side, max_width), dtype='float32')
        mask = mx.nd.zeros((batch_size, 1, short_side, max_width), dtype='float32')
        max_w = 0
        for i, img in enumerate(batch_imgs):
            img_np = cv2.imread(img)
            img_np = self.image_resize(img_np)
            img_np = cv2.cvt_color(img_np, cv2.COLOR_BRG2RGB)
            h, w   = img_np.shape[:2]
            img_nd = transform_fn(img_nd)
            data[i, :, :, :w] = img_nd
            mask[i, :, :, :w] = 1.0
            if w > max_w:
                max_w = w
        data = data.as_in_context(self.ctx)
        mask = mask.as_in_context(self.ctx)
        _, pred2, atten_mask = self.model(img_nd, mask)
        predict = mx.nd.softmax(pred2).asnumpy()
        predict = np.argmax(predict, axis=-1)
        text_list = []
        for i in range(batch_size):
            text = self.ids2text(predict[i])
            text_list.append(text)
        return text_list

    def test_text_line(self, img_dir, save_path):
        file_list = os.listdir(img_dir)
        img_list  = [file for file in file_list if file[-3:] in img_type]
        fi_w = open(save_path, 'w')
        img_list.sort()
        
        for img in img_list:
            img_np = cv2.imread(os.path.join(img_dir, img))
            text = self.single_test(img_np)
            fi_w.write(img + '\t' + text + '\n')

if __name__ == '__main__':           
    tester = Tester()
    img_dir   = '/home/disk0/sw/yolo_crnn/bingli/1239689 系统导出/text_lines'
    save_path = '/home/disk0/sw/yolo_crnn/bingli/1239689 系统导出/recog_result.txt'
    tester.test_text_line(img_dir, save_path)

        
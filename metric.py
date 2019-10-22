#coding=utf-8
from mxnet.metric import EvalMetric
import numpy as np
from mxnet import gluon
import sys

class MyAcc(EvalMetric):
    def __init__(self):
        super(MyAcc, self).__init__('myacc')

    def update(self, preds, labels, mask):
        labels = labels.asnumpy().astype('int32')
        preds = preds.asnumpy()
        preds = np.argmax(preds, axis=-1).astype('int32')
        mask = mask.asnumpy()
        acc = preds == labels
        accuracy = np.sum(acc*mask,axis=-1)/np.sum(mask, axis=-1)
        accuracy = np.mean(accuracy)
        self.sum_metric += accuracy
        self.num_inst += 1

def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y - 1] + 1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + 1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]


def demo(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    list = getStepList(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    result = str("%.2f" % result) + "%"


def load_pred_file(pred_file):
    label_list, pred_list, img_list = [], [], []

    with open(pred_file, 'r', encoding='utf-8') as fi:
        line_list = fi.readlines()
    for line in line_list:
        lst = line.strip().split('|||')
        for i, sen in enumerate(lst[:2]):
            if ' ' in sen:
                sen = sen.split(' ')
            else:
                sen = list(sen)
            if i == 0:
                label_list.append(sen)
            else:
                pred_list.append(sen)
            img_list.append(lst[-1])
    return label_list, pred_list, img_list

def stat_acc(pred_file):
    label_list, pred_list, img_list = load_pred_file(pred_file)
    word_num, dist_num, ins_num, del_num, sub_num = 0, 0, 0, 0, 0
    pred_correct_num = 0
    for i, (label, pred) in enumerate(zip(label_list, pred_list)):
        if len(pred) > 5 * len(label):
            pred = pred[:5 * len(label)]
        d_array = editDistance(label, pred)
        result = getStepList(label, pred, d_array)
        word_num += len(label)
        dist_num += result.count('e')
        ins_num += result.count('i')
        del_num += result.count('d')
        sub_num += result.count('s')

        if len(label) == result.count('e') and len(label) == len(pred):
            pred_correct_num += 1
        # else:
        #     print('line:%d' % i, 'label:%s' % (''.join(label)), 'pred:%s' % (''.join(pred)))
        #     print('ins err:', result.count('i'), 'del err:', result.count('d'),
        #           'sub err:', result.count('s'))
    total_sample = len(label_list)
    word_acc = (word_num - del_num - sub_num) / word_num * 100.0
    sen_acc = pred_correct_num / total_sample * 100.0
    word_precision = (word_num - del_num - sub_num - ins_num) / word_num * 100.0
    print('ins err:', ins_num, 'del err:', del_num, 'sub err:', sub_num)
    print('the number of correct sentence:%.4f%%' % (sen_acc))
    print('the accuracy of word:%.4f%%' % (word_acc))
    print('the precision of word:%.4f%%' % (word_precision))
    return word_acc, word_precision
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        sys.exit()
    stat_acc(sys.argv[1])
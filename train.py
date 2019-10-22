#coding=utf-8
import os
import cv2
import time
import mxnet as mx
import tqdm
import numpy as np
from mxboard import SummaryWriter
from mxnet.gluon.data import DataLoader
from nets.model import AttentionModel
from dataset import FixedSizeDataset, BucketDataset, BucketSampler
from metric import MyAcc, stat_acc
from configs import chr_config as cfg

class Trainer(object):

    def __init__(self):
        if not os.path.exists(cfg.checkpoint):
            os.makedirs(cfg.checkpoint)
        if not os.path.exists(cfg.test_result_dir):
            os.makedirs(cfg.test_result_dir)
        self.ctx = [mx.gpu(int(i)) for i in cfg.gpus.split(',')]
        print('running on:', self.ctx)
        self.model = AttentionModel(cfg.backbone_kwargs, cfg.encoder_kwargs, cfg.decoder_kwargs)
        self.model.hybridize()
        self.model_prefix = os.path.join(cfg.checkpoint, 'atten_model')

        if cfg.load_epoch == 0:
            if cfg.pretrain_base is not None:
                self.model.load_parameters(cfg.pretrain_base, ignore_extra=True,
                                           ctx=self.ctx, load_params='backbone')
                print('load predtrain base model!')
            self.model.initialize(init=mx.init.Xavier(), ctx=self.ctx)
            print('initialize model params!')
        else:
            model_path = '%s-%04d.params'%(self.model_prefix, cfg.load_epoch)
            self.model.load_parameters(model_path, ctx=self.ctx)
            print('load epoch %d params!'%cfg.load_epoch)

        self.train_dataloader, self.val_dataloader = self.get_dataloader(bucktet_mode=cfg.bucket_mode)
        optimizer = mx.optimizer.AdaDelta(learning_rate=cfg.lr, wd=cfg.wd)
        self.trainer  = mx.gluon.Trainer(self.model.collect_params(), optimizer)

        self.ce_loss = mx.gluon.loss.SoftmaxCELoss()
        self.acc_metric = MyAcc()

    def get_dataloader(self, bucktet_mode=False):

        if bucktet_mode:
            train_dataset = BucketDataset(cfg.train_data_path, cfg.voc_path,
                                          short_side=cfg.short_side,
                                          fix_width=cfg.fix_width,
                                          max_len=cfg.max_char_len,
                                          use_augment=True,
                                          add_symbol=True,
                                          max_sample_num=100000,
                                          load_bucket_path='./data/%s_bucket.json'%cfg.dataset_name)
            val_dataset = BucketDataset(cfg.val_data_path, cfg.voc_path,
                                     short_side=cfg.short_side,
                                     fix_width=cfg.fix_width,
                                     max_len=cfg.max_char_len,
                                     use_augment=False,
                                     add_symbol=True,
                                     max_sample_num=10000)
            train_sampler = BucketSampler(cfg.batch_size,
                                         train_dataset.bucket_dict,
                                         shuffle=True,
                                         last_batch='discard')
            val_sampler = BucketSampler(1,
                                        val_dataset.bucket_dict,
                                        shuffle=False,
                                        last_batch='keep')
            train_dataloader = DataLoader(train_dataset,
                                          batch_sampler=train_sampler,
                                          num_workers=cfg.num_workers,
                                          pin_memory=True)
            val_dataloader = DataLoader(val_dataset,
                                        batch_sampler=val_sampler,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True)
        else:
            train_dataset = FixedSizeDataset(cfg.train_data_path, cfg.voc_path,
                                            short_side=cfg.short_side,
                                            fix_width=cfg.fix_width,
                                            max_len=cfg.max_char_len,
                                            use_augment=True,
                                            add_symbol=True,
                                            max_sample_num=100000)
            val_dataset = FixedSizeDataset(cfg.val_data_path, cfg.voc_path,
                                            short_side=cfg.short_side,
                                            fix_width=cfg.fix_width,
                                            max_len=cfg.max_char_len,
                                            use_augment=False,
                                            add_symbol=True,
                                            max_sample_num=10000)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=cfg.batch_size,
                                          last_batch='discard',
                                          shuffle=True,
                                          num_workers=cfg.num_workers,
                                          pin_memory=True)
            val_dataloader   = DataLoader(val_dataset,
                                          batch_size=cfg.batch_size,
                                          last_batch='keep',
                                          num_workers=cfg.num_workers,
                                          pin_memory=True)
        return train_dataloader, val_dataloader

    def train(self, epoch):
        start, disp_time = time.time(), time.time()
        save_model_path  = '%s-%04d.params' % (self.model_prefix, epoch)
        for idx, data in enumerate(self.train_dataloader):
            s_data = mx.gluon.utils.split_and_load(data[0], ctx_list=self.ctx)
            s_mask = mx.gluon.utils.split_and_load(data[1], ctx_list=self.ctx)
            t_label= mx.gluon.utils.split_and_load(data[2], ctx_list=self.ctx)
            t_mask = mx.gluon.utils.split_and_load(data[3], ctx_list=self.ctx)
            l_list = []

            with mx.autograd.record():
                for sd, sm, tl, tm in zip(s_data, s_mask, t_label, t_mask):
                    pred1, pred2, _ = self.model(sd, sm)
                    loss = self.ce_loss(pred1, tl, tm.expand_dims(axis=2)) + \
                           self.ce_loss(pred2, tl, tm.expand_dims(axis=2))
                    l_list.append(loss)
                mx.autograd.backward(l_list)
            self.trainer.step(cfg.batch_size)
            mx.nd.waitall()
            self.acc_metric.update(mx.nd.softmax(pred1), tl, tm)
            self.acc_metric.update(mx.nd.softmax(pred2), tl, tm)
            if idx % cfg.disp_batchs == 0:
                curr_time = time.time()
                acc = self.acc_metric.get()
                cost = mx.nd.mean(loss).asnumpy()[0]
                print('epoch:%d'%(epoch), 'step:%d'%idx, 'loss:%.5f'%cost, 
                      'accuracy:%.4f'%acc[1], 'cost time:%.4f'%(curr_time-disp_time))
                disp_time = curr_time
            if idx % cfg.save_step==0:
                self.model.save_parameters(save_model_path)
        mx.nd.waitall()
        print('epoch:%d, training time: %.1f sec' % (epoch, time.time() - start))
        self.model.save_parameters(save_model_path)
        self.acc_metric.reset()

    def evaluate(self, save_result_path):
        ids2text = self.val_dataloader._dataset.ids2text
        imgs_list= self.val_dataloader._dataset.imgs_list
        fi_w = open(save_result_path, 'w')

        for i, data in enumerate(self.val_dataloader):
            s_data = data[0].as_in_context(self.ctx[0])
            s_mask = data[1].as_in_context(self.ctx[0])
            idxs   = data[-1].asnumpy()
            pred1, pred2, atten_masks = self.model(s_data, s_mask)
            #predict = mx.nd.stack(mx.nd.softmax(pred1), mx.nd.softmax(pred2), axis=0).asnumpy()
            predict = mx.nd.softmax(pred2).asnumpy()
            #predict = np.max(predict, axis=0, keepdims=False)
            predict = np.argmax(predict, axis=-1)
            target  = data[2].asnumpy()
            atten_masks = atten_masks.asnumpy()
            bs, seq_len = predict.shape[:2]
            for j in range(bs):
                idx = idxs[j]
                img_path = imgs_list[idx]
                predi = predict[j]
                targi = target[j]
                pred_text = ids2text(predi)
                targ_text = ids2text(targi)
                text = ''.join(targ_text) + '|||' + ''.join(pred_text) + '|||' + img_path + '\n'
                if cfg.show_atten_map:
                    import pdb
                    pdb.set_trace()
                    self.show_attention_map(img_path, atten_masks, len(pred_text), cfg.save_atten_dir)
                fi_w.write(text)
        fi_w.close()
        return stat_acc(save_result_path)

    def show_attention_map(self, img_path, atten_masks, text_len, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_np  = cv2.imread(img_path)
        img_np  = self.val_dataloader._dataset.image_resize(img_np)
        img_np  = np.float32(img_np)/255.0
        h, w    = img_np.shape[:2]
        atten_masks = np.reshape(atten_masks, (-1, h//4, w//4))
        for idx in range(text_len):
            mask    = atten_masks[idx]
            m_max, m_min = np.max(mask), np.min(mask)
            mask    = (mask-m_min)/m_max
            mask    = cv2.resize(mask, (w, h))
            heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap)/255.0
            cam     = heatmap + np.float32(img_np)
            cam     = cam/np.max(cam)
            cam     = np.uint8(255*cam)
            save_name = 'attenion_mask_%03d.jpg'%idx
            cv2.imwrite(os.path.join(save_dir, save_name), cam)
        return cam

    def run(self):
        start_epoch = cfg.load_epoch + 1
        best_acc = 0.0
        for epoch in range(start_epoch, cfg.num_epochs):
            save_results = os.path.join(cfg.test_result_dir, 'epoch_%03d.txt'%epoch)
            self.train(epoch)
            if epoch % cfg.validate_step == 0:
                val_acc, _ = self.evaluate(save_results)
                if val_acc > best_acc:
                    best_acc = val_acc
                    model_path = '%s-best.params'%(self.model_prefix)
                    self.model.save_parameters(model_path)
            os.remove('%s-%04d.params'%(self.model_prefix, epoch))
            if epoch in cfg.decay_steps:
                self.trainer.set_learning_rate(self.trainer.learning_rate*cfg.lr_decay)
        self.model.load_parameters(model_path, ctx=self.ctx)
        self.model.export(self.model_prefix, 0)

    def test(self, export_model=True):
        model_path  = '%s-%s.params'%(self.model_prefix, 'best')
        self.model.load_parameters(model_path, ctx=self.ctx)
        self.evaluate('best_model_pred.txt')
        if export_model:
            self.model.export(self.model_prefix, 0)

if __name__ == '__main__':
    train = Trainer()
    train.run()
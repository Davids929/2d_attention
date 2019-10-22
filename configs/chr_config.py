#coding=utf-8

# backbone module
backbone_kwargs = {'backbone_type':'resnet',
                   'num_blocks_list':[1, 3, 3, 4, 3],
                   'channels_list':[64, 128, 256, 512, 512],
                   'strides_list':[(2,2), (2,2), (2,2), (1,1), (1,1)],
                   'use_backbone':True}
#densenet 97
# densenet_kwargs = { 'backbone_type':'densenet',
#                     'init_channels':64, 
#                     'growth_rate':32, 
#                     'num_layers_list':[6, 12, 12, 16], 
#                     'strides_list':[(2,2), (2,2), (1,1)], 
#                     'bn_size':4, 
#                     'dropout':0.1}

#relation attention module
ra_kwargs = {'num_layers':2,
             'hidden_size':1024,
             'output_size':512,
             'num_heads':8,
             'dropout':0.1}
#parallel attention module
max_char_len = 60
pa_kwargs = {'hidden_size':512, 'output_size':max_char_len}
# RNN module
rnn_kwargs      = None
encoder_kwargs  = {'ra_kwargs':ra_kwargs,
                   'pa_kwargs':pa_kwargs,
                   'rnn_kwargs':rnn_kwargs}
# decoder
voc_size = 6053+4
#decoder relation attention module
decoder_kwargs   = {'voc_size':voc_size, 'ra_kwargs':ra_kwargs}

# dataset
bucket_mode     = True
dataset_name    = 'receipt'
data_augment    = True
short_side      = 32
fix_width       = None
train_data_path = ['/home/disk0/sw/data/receipts/train_lines.txt',
                   '/home/disk0/sw/data/receipts/baidu_lines.txt']
val_data_path   = '/home/disk0/sw/data/receipts/val_lines.txt'
# train_data_path = '/home/disk0/sw/data/synth_data/train/labels.txt'
# val_data_path   = '/home/disk0/sw/data/synth_data/test/labels.txt'
voc_path        = './data/char_std_5990.txt'

#hyperparameter
num_workers = 8
batch_size  = 64
gpus        = '0'
wd          = 0.0001
lr          = 0.01
decay_steps = [20, 30]
lr_decay    = 0.1

test_result_dir = './test_results/' + dataset_name
checkpoint      = './checkpoint/' + dataset_name
pretrain_base   = './checkpoint/synth/atten_model-best.params'
load_epoch      = 0
num_epochs      = 40
save_step       = 2000
validate_step   = 2
disp_batchs     = 10
show_atten_map  = False
save_atten_dir  = test_result_dir  + '/attention_map'
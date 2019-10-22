#coding=utf-8

# model parameters
# backbone module
backbone_kwargs = {'num_blocks_list':[1, 3, 3, 4, 3],
                   'channels_list':[64, 128, 256, 512, 512],
                   'strides_list':[(2,2), (2,2), (2,2), (1,1), (1,1)],
                   'use_backbone':True}
#relation attention module
ra_kwargs = {'num_layers':2,
             'hidden_size':512,
             'output_size':128,
             'num_heads':4,
             'dropout':0.1}
#parallel attention module
max_char_len = 25
pa_kwargs    = {'hidden_size':256,
                'output_size':max_char_len}
# RNN module
rnn_kwargs      = None
encoder_kwargs  = {'ra_kwargs':ra_kwargs,
                   'pa_kwargs':pa_kwargs,
                   'rnn_kwargs':None}
# decoder
voc_size = 36 + 4
#decoder relation attention module
decode_ra_kwargs = {'num_layers':2,
                    'hidden_size':512,
                    'output_size':256,
                    'num_heads':4,
                    'dropout':0.1}
decoder_kwargs   = {'voc_size':voc_size,
                    'ra_kwargs':decode_ra_kwargs}

# dataset
bucket_mode     = True
dataset_name    = 'icdar2015'
data_augment    = True
short_side      = 32
fix_width       = 100
train_data_path = './data/icdar2015/train_data.txt'
val_data_path   = './data/icdar2015/test_data.txt'
# train_data_path = './data/synth90k/synth_train.txt'
# val_data_path   = './data/synth90k/synth_val.txt'
voc_path        = './data/eng_voc_dict.txt'

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
pretrain_base   = checkpoint + '/' + 'atten_model-best.params'
load_epoch      = 0
num_epochs      = 40
save_step       = 2000
validate_step   = 1
disp_batchs     = 10
show_atten_map  = False
save_atten_dir  = test_result_dir + '/' + 'attention_map'
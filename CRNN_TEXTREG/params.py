from CRNN_TEXTREG.tool import alphabets
alphabet = alphabets.alphabet
keep_ratio = False
manualSeed = 1234
random_sample = True
height = 32
width = 128
number_hidden = 256
n_channel = 1

pretrained = ''
expr_dir = 'expr'
dealwith_lossnan = False

cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 1 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0 # number of data loading workers

# training process
displayInterval = 100 # interval to be print the train loss
valInterval = 1000 # interval to val the model loss and accuray
saveInterval = 1000 # interval to save model
n_val_disp = 10 # number of samples to display when val the model

# finetune
nepoch = 1000 # number of epochs to train for
batchSize = 2 # input batch size
lr = 0.0001 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
adam = False # whether to use adam (default is rmsprop)
adadelta = False # whether to use adadelta (default is rmsprop)
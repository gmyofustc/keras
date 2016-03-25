# -*- coding: utf-8 -*-
from __futre__ import print_function
import numpy as np
np.random.seed(1234)	#fix the random seed

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Merge
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from new_modules.self_defined_modules import ComputeCosinSimilarity

batchsize = 4
nb_epoch = 50
max_sent_length = 40
max_sent_num = 25
max_txt_length = 1000
max_pos_length = 40
max_neg_length = 40
max_ques_length = 40
vocab_size = xxx
emb_dim = 50

model = Graph()
emb_file = './embeddings.txt'
emb_shared = LoadEmbedding(emb_file)

model.add_input(name = 'txt_info', batch_input_shape = (batchsize, max_txt_length), dtype = 'int')
model.add_input(name = 'ques_info', batch_input_shape = (batchsize, max_ques_length), dtype = 'int')
model.add_input(name = 'pos_info', batch_input_shape = (batchsize, max_pos_length), dtype = 'int')
model.add_input(name = 'neg_info', batch_input_shape = (batchsize, max_neg_length), dtype = 'int')
model.add_input(name = 'sent_mask', batch_input_shape = (batchsize, max_sent_num), dtype = 'float')

model.add_shared_node(Embedding(vocab_size, emb_dim, mask_zero = True, weights = emb_shared), name = 'Shared_Embedding', inputs = ['txt_info', 'ques_info', 'pos_info', 'neg_info'], outputs = ['txt_emb', 'ques_emb', 'pos_emb', 'neg_emb'])
model.add_node(Reshape((batchsize*max_sent_num, max_sent_length, emb_dim, 1)), name = 'Txt_Emb_Reshape', input = 'txt_emb')
model.add_node(Reshape((batchsize, max_sent_length, emb_dim, 1)), name = 'Ques_Emb_Reshape', input = 'ques_emb')
model.add_node(Reshape((batchsize, max_sent_length, emb_dim, 1)), name = 'Pos_Emb_Reshape', input = 'pos_emb')
model.add_node(Reshape((batchsize, max_sent_length, emb_dim, 1)), name = 'Neg_Emb_Reshape', input = 'neg_emb')

model.add_node(Permute((2, 1, 3)), name = 'Txt_Emb_Permute', input = 'Txt_Emb_Reshape')
model.add_node(Permute((2, 1, 3)), name = 'Ques_Emb_Permute', input = 'Ques_Emb_Reshape')
model.add_node(Permute((2, 1, 3)), name = 'Pos_Emb_Permute', input = 'Pos_Emb_Reshape')
model.add_node(Permute((2, 1, 3)), name = 'Neg_Emb_Permute', input = 'Neg_Emb_Reshape')


sentence_level_filters = emb_dim*2
sentence_conv_row = 3
sentence_conv_col = 1
sent_pool_size = (40, 1)
model.add_shared_node(Convolution2D(sentence_level_filters, sentence_conv_row, sentence_conv_col, activation = 'relu', border_mode = 'same'), name = 'Shared_Sentence_Convolution', inputs = ['Txt_Emb_Permute', 'Ques_Emb_Permute', 'Pos_Emb_Permute', 'Neg_Emb_Permute'], outputs = ['txt_sent_conv', 'ques_sent_conv', 'pos_sent_conv', 'neg_sent_conv'])
model.add_shared_node(MaxPooling2D(sent_pool_size), name = 'Sent_Max_Pool', inputs = ['txt_sent_conv', 'ques_sent_conv', 'pos_sent_conv', 'neg_sent_conv'], outputs = ['txt_sent_pool', 'ques_sent_pool', 'pos_sent_pool', 'neg_sent_pool'])

txt_sent_level_shape = (batchsize, max_sent_num, sentence_level_filters)
ques_sent_level_shape = (batchsize, 1, sentence_level_filters)
pos_sent_level_shape = (batchsize, 1, sentence_level_filters)
neg_sent_level_shape = (batchsize, 1, sentence_level_filters)
model.add_node(Reshape(txt_sent_level_shape), name = 'txt_sents', input = 'txt_sent_pool')
model.add_node(Reshape(ques_sent_level_shape), name = 'ques_sent', input = 'ques_sent_pool')
model.add_node(Reshape(pos_sent_level_shape), name = 'pos_sent', input = 'pos_sent_pool')
model.add_node(Reshape(neg_sent_level_shape), name = 'neg_sent', input = 'neg_sent_pool')

sent_max_pool = 3
model.add_node(ComputeCosinSimilarity(), name = 'Txt_Ques_Similarity', inputs = ['txt_sent_pool', 'ques_sent_pool'], merge_mode = 'concat', concat_axis = 1)
model.add_node(Merge(['Txt_Ques_Similarity', 'sent_mask']), name = 'Mask_Txt_Ques_Similarity', mode = 'mul')
model.add_node(Permute((1, 2, 'x')), name = 'Mask_Txt_Ques_Similarity_Permute', input = 'Mask_Txt_Ques_Similarity')
model.add_node(KMaxPool(sent_max_pool), name = 'KMax_Sent', inputs = ['Mask_Txt_Ques_Similarity_Permute', 'txt_sents'], merge_mode = 'concate', concat_axis = -1)

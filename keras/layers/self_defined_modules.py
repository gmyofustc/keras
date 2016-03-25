# -*- coding: utf-8 -*-
import keras.backend as K
import theano
import theano.tensor as T
from keras.layers.core import Layer

class ComputeCosinSimilarity(Layer):
	def __init__(self, **kwargs):
		super(ComputeCosinSimilarity, self).__init__(**kwargs)

	@property
	def output_shape(self):
		time_step = (self.input_shape[1] - 1)
		return (self.input_shape[0], time_step)

	def get_output(self, train = False):
		X = self.get_input(train)
		X_txt = X[:,:-2,:]
		X_qa = X[:,-2:-1,:]
		X_qa = X_qa.reshape((X_qa.shape[0], X_qa.shape[2]))
		txt_qa_similarities = self.compute_similarity(X_txt, X_qa)
		return txt_qa_similarites

	def get_config(self):
		config = {'name' = self.__class__.__name__}
		base_config = super(ComputeCosinSimilarity, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


	######################specific for this class only######################	
	def compute_similarity(self, txt, qa):
		#txt shape (batchsize, tiem_step, ndim) ==> (tiem_step, batchsize, ndim)
		(batchsize, time_step, ndim) = txt.shape
		txt = txt.dimshuffle(1, 0, 2)
		similarities, updates = theano.scan(self._step,
						    				sequences = [txt],
						    				non_sequences = [qa]
										   )
		#similarities shape (time_step, batchsize)==>(batchsize, time_step)
		similarities = similarities.reshape((time_step, batchsize))
		similarities = similarities.dimshuffle(1, 0)
		return similarities
		
	def _step(self, txt, qa):
		(batchsize, ndim) = txt.shape
		#compute the L2 norm of txt and qa
		txt_norm = txt.norm(2, axis = -1)
		qa_norm = qa.norm(2, axis = -1)
		txt = txt.reshape((batchsize, 1, ndim))
		qa = txt.reshape((batchsize, ndim, 1))
		txt_qa_dot = T.batched_dot(txt, qa)
		txt_qa_dot = txt_qa_dot.reshape((batchsize, 1))
		#the cosin similarity between two tensors
		txt_qa_similarity = txt_qa_dot/txt_norm*qa_norm
		return txt_qa_similarity

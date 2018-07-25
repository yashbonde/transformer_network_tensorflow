'''
Your knowledge is only overshadowed by your stupidity, Starscrem. - Megatron

# transformer network
'''

# importing the required dependencies
import numpy as np # matrix math
import tensorflow as tf # ML

class Transformer(object):
	"""docstring for TransformerNetwork"""
	def __init__(self, VOCAB_SIZE, scope = 'transformer_network', H_V = 8, D_MODEL = 512, D_K = 64, D_V = 64,
		FF_MID = 2014, NUM_STACKS_E = 3, NUM_STACKS_D = 3, WARMUP_STEPS = 4000):
		# constants
		self.VOCAB_SIZE = VOCAB_SIZE # for final output
		self.H_V = H_V # number of heads in multihead attention mechanism
		self.D_MODEL = D_MODEL # input size
		self.D_K = D_K # self.D_MODEL//self.H_V 
		self.D_V = D_V # self.D_MODEL//self.H_V
		self.FF_MID = FF_MID # number of units in middle layer of fully connected stack
		self.NUM_STACKS_E = NUM_STACKS_E # number of stacks in encoder module
		self.NUM_STACKS_D = NUM_STACKS_D # number of stacks in decoder module
		self.STEP = 0 # the global steps
		self.WARMUP_STEPS = WARMUP_STEPS # this is the initial number of steps for defining the learning rate

		# scope
		self.scope = scope

		# dynamics
		self.embedding_dim = D_MODEL

		# time varying values
		self.curr_step = 0
		self.lr = min(1/np.sqrt(self.curr_step), self.curr_step/np.power(self.WARMUP_STEPS, 1.5))/np.sqrt(self.D_MODEL)

		# initialize the network
		self.input_placeholder = tf.placeholder(tf.float32, [None, self.D_MODEL]) # input
		self.output_placeholder = tf.placeholder(tf.float32, [None, self.D_MODEL]) # output
		self.labels_placeholder = tf.placeholder(tf.float32, [None, self.VOCAB_SIZE]) # vocab converted to one-hot
		self.position = tf.placeholder(tf.int32, [None])
		self.make_transformer()
		self.build_loss()

	def _masked_multihead_attention(self, V, K, Q, reuse, back_side = False):
		# this is the meachnais tham happens, whenever we try to connect the output at further time step to
		# the ones behind, this is only used in this case. Otherwise you can use the simple multihead attention
		# inplace of this.

		# since the way I am going to implement the data I won't need this functionality
		# so for now we can just return the multihead attention
		if not back_side:
			with tf.variable_scope('masked_multihead_attention'):
				return self.multihead_attention(V, K, Q, reuse)

	def multihead_attention(self, V, K, Q, reuse):
		with tf.variable_scope('multihead_attention', reuse = reuse):
			head_tensors = [] # list to store all the output of heads
			for i in range(self.H_V):
				# iterate over all the linear layers
				if not reuse:
					# this is the first time it has been called
					# common sense tells that if we are not reusing the outer loop then it is first time
					# since all the projection weight for linear networks are different
					reuse_linear = False
				else:
					reuse_linear = True
				with tf.variable_scope('linear' + str(i), reuse = reuse_linear):
					# weight value
					weight_q = tf.get_variable('weight_q_l' + str(i), (self.D_MODEL, self.D_K), initializer = tf.truncated_normal_initializer)
					weight_k = tf.get_variable('weight_k_l' + str(i), (self.D_MODEL, self.D_K), initializer = tf.truncated_normal_initializer)
					weight_v = tf.get_variable('weight_v_l' + str(i), (self.D_MODEL, self.D_V), initializer = tf.truncated_normal_initializer)
					# projected values
					k_proj = tf.matmul(K, weight_k)
					q_proj = tf.matmul(Q, weight_q)
					v_proj = tf.matmul(V, weight_v)

					# Scale Dot Product Attention
					qkt = tf.matmul(q_proj, v_proj, transpose_b = True)
					qkt /= np.sqrt(self.D_K)
					soft_qkt = tf.nn.softmax(qkt)
					head = tf.matmul(soft_qkt, v_proj)

					# add the new
					head_tensors.append(head)

			# now we proceed to the concatenation
			head_concat = tf.reshape(tf.stack(head_tensors), [-1, self.D_MODEL])
			mha_out_weight = tf.get_variable('mha_ow', shape = [head_concat.shape[-1], self.D_MODEL])
			return tf.matmul(head_concat, mha_out_weight)

	def _encoder(self, enc_in):
		'''
		This is the encoder module, it takes input the input values combined with positional encodings
		'''
		with tf.variable_scope('encoder'):
			stack_op_tensors = [] # list of stack output tensors
			for i in range(self.NUM_STACKS_E):
				# set the stack_in
				if i == 0:
					reuse_stack = False
					reuse_global = False
					stack_in = enc_in
				else:
					reuse_global = True
					reuse_stack = True
					stack_in = stack_op_tensors[-1]

				with tf.variable_scope('stack', reuse = reuse_stack):
					# sublayer 1 operations
					mha_out = self.multihead_attention(V = stack_in, K = stack_in, Q = stack_in, reuse = reuse_global) + stack_in
					norm_1_out = tf.nn.l2_normalize(mha_out)
					norm_1_out = tf.nn.dropout(norm_1_out, keep_prob = 0.9)

					# sublayer 2 operations
					ff_mid = tf.layers.dense(norm_1_out, self.FF_MID, activation = tf.nn.relu, use_bias = False)
					ff_out = tf.layers.dense(ff_mid, self.D_MODEL, activation = tf.nn.relu, use_bias = False)
					ff_out += norm_1_out
					ff_out_norm = tf.nn.l2_normalize(ff_out)

					# add to the stack outputs
					stack_op_tensors.append(ff_out)

		# we return these tensors as this is what we feed to the decoder module
		return stack_op_tensors

	def _decoder(self, dec_in, encoder_op_tensors):
		'''
		Decoder module takes input the outputs combined with the positional encodings
		'''
		with tf.variable_scope('decoder'):
			stack_op_tensors = []
			for i in range(self.NUM_STACKS_D):
				# for each stack
				if i == 0:
					reuse_stack = False
					reuse_global = False
					stack_in = dec_in
				else:
					reuse_global = True
					reuse_stack = True
					stack_in = stack_op_tensors[-1]

				with tf.variable_scope('stack', reuse = reuse_stack):
					# for sublayer 1 operations
					masked_mha_op = self._masked_multihead_attention(stack_in, stack_in, stack_in, reuse = reuse_global) + dec_in
					mmha_op_norm = tf.nn.l2_normalize(masked_mha_op) # normalised masked multi head norm
					mmha_op_norm = tf.nn.dropout(mmha_op_norm, keep_prob = 0.9)

					# for sublayer 2 operations
					mha_op = self.multihead_attention(V = encoder_op_tensors[i], K = encoder_op_tensors[i], Q = mmha_op_norm, reuse = reuse_global)
					mha_op += mmha_op_norm
					mha_op_norm = tf.nn.l2_normalize(mha_op)
					mha_op_norm = tf.nn.dropout(mha_op_norm, keep_prob = 0.9)

					# for sublayer 3 operations
					ff_mid = tf.layers.dense(mha_op_norm, self.FF_MID, activation = tf.nn.relu, use_bias = False)
					ff_out = tf.layers.dense(ff_mid, self.D_MODEL, activation = tf.nn.relu, use_bias = False)
					ff_out += mha_op_norm
					ff_out_norm = tf.nn.l2_normalize(ff_out)

					# add to the stack outputs
					stack_op_tensors.append(ff_out_norm)

			# once the stacks are finished we can then do linear and softmax
			stacks_out = stack_op_tensors[-1]
			decoder_op = tf.layers.dense(stacks_out, self.VOCAB_SIZE, activation = tf.nn.softmax)

			return decoder_op

	def make_transformer(self):
		with tf.variable_scope(self.scope):
			# stack
			encoder_stack_op = self._encoder(self.input_placeholder)
			self.decoder_op = self._decoder(self.output_placeholder, encoder_stack_op)

	def build_loss(self):
		# calculating the loss
		self.loss = tf.losses.softmax_cross_entropy(labels = self.labels_placeholder, logits = self.decoder_op, label_smoothing = 0.1)
		# getting the training step
		self.train_step = tf.train.AdamOptimizer(learning_rate = self.lr, beta2 = 0.98, epsilon = 1e-09).minimize(self.loss)

	def initialize_network(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initilizer())

	#### PUBLIC FUNCITONS ####
	def get_variables(self):
		# return the list of all the trianable variables
		trainable_ops = tf.trainable_variables(scope = 'transformer_network')
		return trainable_ops

	def run(self, inputs, outputs, seqlen, return_sequences = False):
		'''
		Run a single iteration of the transformer cell
		Args:
			inputs: the input values to transformer
			outputs: the output value to the transformer network
			seqlen: length of sequence
			return_sequences: if True return the total array to the 
		'''
		transformer_output = []
		for i in range(seqlen):
			feed = {self.input_placeholder: inputs[i], self.output_placeholder: outputs[i]}
			tt_o = self.sess.run(self.decoder_op, feed_dict = feed)
			transformer_output.append(tt_o)
		if return_sequences:
			return transformer_output
		else:
			return transformer_output[-1]

	def train_network(self, inputs, outputs, labels, curr_epoch):
		self.curr_step = curr_epoch
		if self.curr_step == 0:
			self.initialize_network()
		self.lr = min(1/np.sqrt(self.curr_step), self.curr_step/np.power(self.WARMUP_STEPS, 1.5))/np.sqrt(self.D_MODEL)
		# both inputs and outputs already have the positional encoding added to them
		feed_dict = {self.input_placeholder: inputs, self.output_placeholder: outputs, self.labels_placeholder: labels}
		loss, _ = sess.run([self.loss, self.train_step], feed_dict = feed_dict)

		return loss

	def save_model():
		# save the model as a frozen graph
		pass













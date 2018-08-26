'''
Your knowledge is only overshadowed by your stupidity, Starscrem. - Megatron

# transformer network
'''

LOWEST_VAL = 1e-36

# importing the required dependencies
import numpy as np # matrix math
import tensorflow as tf # ML

class Transformer(object):
	"""docstring for TransformerNetwork"""
	def __init__(self, VOCAB_SIZE, scope = 'transformer_network', NUM_HEADS = 8, DIM_MODEL = 512,
		FF_MID = 2048, NUM_STACKS_E = 3, NUM_STACKS_D = 3, WARMUP_STEPS = 4000):
		# constants
		self.VOCAB_SIZE = VOCAB_SIZE # for final output
		self.NUM_HEADS = NUM_HEADS # number of heads in multihead attention mechanism
		self.DIM_MODEL = DIM_MODEL # input size / embedding dimension
		self.DIM_KEY = int(DIM_MODEL/NUM_HEADS) # self.DIM_MODEL//self.NUM_HEADS 
		self.DIM_VALUE = int(DIM_MODEL/NUM_HEADS) # self.DIM_MODEL//self.NUM_HEADS
		self.FF_MID = FF_MID # number of units in middle layer of fully connected stack
		self.NUM_STACKS_E = NUM_STACKS_E # number of stacks in encoder module
		self.NUM_STACKS_D = NUM_STACKS_D # number of stacks in decoder module
		self.STEP = 0 # the global steps
		self.WARMUP_STEPS = WARMUP_STEPS # this is the initial number of steps for defining the learning rate

		# scope
		self.scope = scope

		# dynamics
		self.embedding_dim = DIM_MODEL
		self.generated_length =  tf.placeholder(tf.int32, [1])# length the sequence has been generated, used in masked multi-head attention

		# getting a lot of issues, so the trick is to convert the embedding matrix to a placeholder which can be fed during practice
		# self.masking_matrix = np.ones([12, 12], dtype = np.float32)
		self.masking_matrix = tf.placeholder(tf.float32, [None, None], name = 'embedding_matrix')

		# time varying values
		self.curr_step = 0
		self.lr = min(1/np.sqrt(self.curr_step + 1), self.curr_step/np.power(self.WARMUP_STEPS, 1.5))/np.sqrt(self.DIM_MODEL)

		# initialize the network
		self.input_placeholder = tf.placeholder(tf.float32, [None, self.DIM_MODEL], name = 'input_placeholder') # input
		self.output_placeholder = tf.placeholder(tf.float32, [None, self.DIM_MODEL], name = 'output_placeholder') # output
		self.labels_placeholder = tf.placeholder(tf.float32, [None, self.VOCAB_SIZE], name = 'labels_placeholder') # vocab converted to one-hot
		self.position = tf.placeholder(tf.int32, [None], name = 'position')
		self.make_transformer()
		self.build_loss()

	def _masked_multihead_attention(self, Q, K, V, reuse):
		# this is the place where masking happens, whenever we try to connect the output at further time step to
		# the ones behind, this is only used in this case. Otherwise you can use the simple multihead attention
		# inplace of this.

		# masking values
		self.mask_value = tf.nn.embedding_lookup(self.masking_matrix, self.generated_length) # perform lookup from masking table

		# normalisation value for multihead attention
		root_dk = np.sqrt(self.DIM_KEY)

		# code
		with tf.variable_scope('masked_multihead_attention', reuse = reuse):
			head_tensors = []
			for i in range(self.NUM_HEADS):
				# iterate over all linear layers
				if not reuse:
					# this is the first time this has been called
					reuse_linear = False
				else:
					reuse_linear = True
				with tf.variable_scope('linear' + str(i), reuse = reuse_linear):
					# weight value
					weight_q = tf.get_variable('weight_q_l' + str(i), (self.DIM_MODEL, self.DIM_KEY), initializer = tf.truncated_normal_initializer)
					weight_k = tf.get_variable('weight_k_l' + str(i), (self.DIM_MODEL, self.DIM_KEY), initializer = tf.truncated_normal_initializer)
					weight_v = tf.get_variable('weight_v_l' + str(i), (self.DIM_MODEL, self.DIM_VALUE), initializer = tf.truncated_normal_initializer)
					
					# projected values
					k_proj = tf.matmul(K, weight_k, name = 'k_proj')
					q_proj = tf.matmul(Q, weight_q, name = 'q_proj')
					v_proj = tf.matmul(V, weight_v, name = 'v_proj')

					# scale dot product attention with masking
					qkt = tf.matmul(q_proj, k_proj, transpose_b = True)
					qkt_div = tf.divide(qkt, root_dk)
					qkt_masked = qkt_div * self.mask_value
					soft_qkt = tf.nn.softmax(qkt_masked)
					head = tf.matmul(soft_qkt, v_proj)

					# add this new head to the head list
					head_tensors.append(head)

			# now we proceed to the concatenation
			head_concat = tf.reshape(tf.stack(head_tensors), [-1, self.DIM_MODEL])
			mmha_out_weight = tf.get_variable('mha_ow', shape = [head_concat.shape[-1], self.DIM_MODEL])
			mmha_out = tf.matmul(head_concat, mmha_out_weight)
			return mmha_out


	def multihead_attention(self, V, K, Q, reuse):
		with tf.variable_scope('multihead_attention', reuse = reuse):
			head_tensors = [] # list to store all the output of heads
			for i in range(self.NUM_HEADS):
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
					weight_q = tf.get_variable('weight_q_l' + str(i), (self.DIM_MODEL, self.DIM_KEY), initializer = tf.truncated_normal_initializer)
					weight_k = tf.get_variable('weight_k_l' + str(i), (self.DIM_MODEL, self.DIM_KEY), initializer = tf.truncated_normal_initializer)
					weight_v = tf.get_variable('weight_v_l' + str(i), (self.DIM_MODEL, self.DIM_VALUE), initializer = tf.truncated_normal_initializer)
					
					# projected values
					k_proj = tf.matmul(K, weight_k, name = 'k_proj')
					q_proj = tf.matmul(Q, weight_q, name = 'q_proj')
					v_proj = tf.matmul(V, weight_v, name = 'v_proj')

					# Scale Dot Product Attention
					qkt = tf.matmul(q_proj, k_proj, transpose_b = True)
					qkt /= np.sqrt(self.DIM_KEY)
					soft_qkt = tf.nn.softmax(qkt)
					head = tf.matmul(soft_qkt, v_proj)

					# add the new
					head_tensors.append(head)

			# now we proceed to the concatenation
			head_concat = tf.reshape(tf.stack(head_tensors), [-1, self.DIM_MODEL])
			mha_out_weight = tf.get_variable('mha_ow', shape = [head_concat.shape[-1], self.DIM_MODEL])
			mha_out = tf.matmul(head_concat, mha_out_weight)
			return mha_out


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
					ff_out = tf.layers.dense(ff_mid, self.DIM_MODEL, activation = tf.nn.relu, use_bias = False)
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
					ff_out = tf.layers.dense(ff_mid, self.DIM_MODEL, activation = tf.nn.relu, use_bias = False)
					ff_out += mha_op_norm
					ff_out_norm = tf.nn.l2_normalize(ff_out)

					# add to the stack outputs
					stack_op_tensors.append(ff_out_norm)

			# once the stacks are finished we can then do linear and softmax
			stacks_out = stack_op_tensors[-1]
			print('stacks_out:', stacks_out)
			decoder_op = tf.layers.dense(stacks_out, self.VOCAB_SIZE, activation = tf.nn.softmax)

			return decoder_op

	def make_transformer(self):
		with tf.variable_scope(self.scope):
			# stack
			self.encoder_stack_op = self._encoder(self.input_placeholder)
			self.decoder_op = self._decoder(self.output_placeholder, self.encoder_stack_op)

	def build_loss(self):
		# calculating the loss
		self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.labels_placeholder, logits = self.decoder_op, label_smoothing = 0.1)
		# getting the training step
		self.train_step = tf.train.AdamOptimizer(learning_rate = self.lr, beta2 = 0.98, epsilon = 1e-09).minimize(self.loss)

	def initialize_network(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	#### PUBLIC FUNCITONS ####
	def get_variables(self):
		# return the list of all the trianable variables
		trainable_ops = tf.trainable_variables(scope = 'transformer_network')
		return trainable_ops

	def run(self, input_seq, output_seq, onehot_label, is_training = True):
		'''
		Run a single sentence in transformer
		Args:
			intput_seq: the input vector of shape [<Num_words>, E_dim]
			output_seq: the output value to the transformer network [<Num_words>, E_dim]
			return_sequences: if True return the total array to the 
		'''
		# sanity checks
		if len(input_seq) != len(output_seq):
			raise ValueError('Length of input and output should be equal. Got in: {0}, out: {1}'.format(len(input_seq), len(output_seq)))

		# vars
		transformer_output = []
		seqlen = len(input_seq)
		# input_seq = None --> # get the input embeddings here

		# making the masking lookup table for this sequence
		masking_matrix = np.array([([1,] * (i+1)) + ([LOWEST_VAL,] * (seqlen - i - 1)) for i in range(seqlen)], dtype = np.float32)

		# variables to make if we are training
		if is_training:
			seq_loss = []

		# run over the seqlen
		for i in range(seqlen):
			# prepare the data for this sequence
			feed = {self.input_placeholder: input_seq,
				self.output_placeholder: output_seq,
				self.labels_placeholder: onehot_label,
				self.generated_length: [i],
				self.masking_matrix: masking_matrix}
			
			# run the model for that input
			tt_o = self.sess.run(self.decoder_op, feed_dict = feed)[0]
			transformer_output.append(np.argmax(tt_o))

			# if training is to be done
			if is_training:
				seq_curr_loss, _ = self.sess.run([self.loss, self.train_step], feed_dict = feed)
				seq_loss.append(seq_curr_loss)

		# return the sequential output of the model
		if is_training:
			return transformer_output, seq_loss

		# otherwise
		return transformer_output

	def save_model():
		# save the model as a frozen graph
		pass

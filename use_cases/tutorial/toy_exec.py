# making toy dataset for training transformer network

# importing the dependencies
import pickle
import numpy as np

# some constants for the toy dataset
MAXLEN_SENT_L1 = 100 # maximum length of a sentence for langauge 1
MAXLEN_SENT_L2 = 115 # maximum length of a sentence for langauge 2
VOCAB_SIZE_L1 = 2048 # vocabulary for language 1
VOCAB_SIZE_L2 = 3012 # vocabulary for language 2
num_examples = 10000 # number of samples
test_split = 0.1 # test/total_data

# make the data

# making random lenght sentences
data_l1 = np.asarray([np.random.randint(VOCAB_SIZE_L1, size = np.random.randint(MAXLEN_SENT_L1)) for _ in range(num_examples)])
data_l2 = np.asarray([np.random.randint(VOCAB_SIZE_L2, size = np.random.randint(MAXLEN_SENT_L2)) for _ in range(num_examples)])
# final set
data_l1_final = []
data_l2_final = []
for i in range(num_examples):
    # 9999 is the break value, once we get that as the prediction, break the while loop
    data_l1_final.append(np.append(data_l1[i], VOCAB_SIZE_L1)) # since the stop number is never included
    data_l2_final.append(np.append(data_l2[i], VOCAB_SIZE_L2)) # since the stop number is never included
data_l1_final = np.array(data_l1_final)
data_l2_final = np.array(data_l2_final)

# train the model
tt = TransformerNetwork(VOCAB_SIZE = VOCAB_SIZE_L2, scope = 'toy_transformer',
	H_V = 4, D_MODEL = 128, D_K = 32, D_V = 32, FF_MID = 1000,
	NUM_STACKS_E = 3, NUM_STACKS_D = 3, WARMUP_STEPS = 300)

# clean the data and start training
num_epochs = 1000
batch_size = 32
for i in range(num_epochs):
	data_x, data_y, labels = get_batch()
	loss = tt.train_network(inputs = x_, outputs = y_, labels = labels, curr_epoch = i)
	print('Epoch:{0}, loss:{1}'.format(i, loss))
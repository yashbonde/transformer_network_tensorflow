'''
Utils file for transformer network parameters
Built by Yash Bonde for team msaic_yash_sara_kuma_6223

Cheers!
'''

import numpy as np

# classes
class DatasetManager():
    '''
    Simple class for dataset handling, it has simple built in functions
    '''
    def __init__(self, d1, d2, d3, val_split = 0.1):
        # check of proper shape
        assert d1.shape[0] == d2.shape[0] == d3.shape[0]
        
        self.num_samples = d1.shape[0]
        self.s_idx = 0
        self.e_idx = 0
        
        self.num_iterations = 0
        self.val_split = val_split

        # using reservoir sampling algorithm
        k = int(len(d1) * val_split)
        self.val_idx = np.arange(k)
        for i in range(k, len(d1)):
            j = np.random.randint(i)
            if j < k:
                self.val_idx[j] = i

    '''
    def __del__(self):
        # delete the datasets 
        del self.d1, self.d2, self.d3
    '''

    def get_num_batches(self, batch_size):
        num_batches = int(self.num_samples/batch_size)
        return num_batches

    def get_val_batch(self, d1, d2, d3, batch_size, loop_over = True):
        
        
    def get_batch(self, d1, d2, d3, batch_size, loop_over = True):
        '''
        Get data of batch_size
        '''
        if self.s_idx + batch_size > len(d1):
            
            # this is the case for last turn
            batch_d1 = d1[self.s_idx:]
            batch_d2 = d2[self.s_idx:]
            batch_d3 = d3[self.s_idx:]

            # add leftovers
            leftover_values = batch_size - self.num_samples + self.s_idx
            # print(leftover_values)

            batch_d1 = np.append(batch_d1, d1[:leftover_values])
            batch_d2 = np.append(batch_d2, d2[:leftover_values])
            batch_d3 = np.append(batch_d3, d3[:leftover_values])

            '''
            print(batch_d1.shape)
            print(batch_d2.shape)
            print(batch_d3.shape)
            '''
                        
            # update values
            self.s_idx = 0
            self.e_idx = 0
            self.num_iterations += 1
            return batch_d1, batch_d2, batch_d3
        
        else:
            self.e_idx += batch_size
            batch_d1 = d1[self.s_idx: self.e_idx]
            batch_d2 = d2[self.s_idx: self.e_idx]
            batch_d3 = d3[self.s_idx: self.e_idx]
            self.s_idx += batch_size
            self.num_iterations += 1
            
            # checks
            if self.s_idx == len(d1):
                self.s_idx = 0
                self.e_idx = 0
            
            return batch_d1, batch_d2, batch_d3

    def get_stats(self):
        # return values
        return self.num_iterations


# functions
def load_numpy_array(filepath):
    '''
    return loaded numpy array
    '''
    return np.load(filepath)

def add_padding(inp, pad_id, seqlen):
    '''
    Pad the input values to the required length
    Args:
        inp: input sequence with shape (batch_size, <variable>)
        pad_id: (int) ID for padding element
    Returns:
        (batch_size, seqlen)
        NOTE: zeros are added at the starting
    '''
    sequences = []
    for s in inp:
        if seqlen > len(s):
            s = np.append(arr = np.ones(seqlen - len(s)) * pad_id, values = s)
        else:
            s = s[:seqlen]

        # check 
        if len(s) != seqlen:
            print('THE BUG IS FUCKING HERE...')
            raise SystemExit

        if len(sequences) == 0:
            sequences = np.array([s])
            continue
        
        sequences = np.vstack((sequences, s))
        
    return sequences


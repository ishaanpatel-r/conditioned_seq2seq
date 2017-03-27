import numpy as np
from random import sample


'''
 split data into train (70%), test (15%) and valid(15%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )

'''
def split_dataset(data, ratio = [0.7, 0.1, 0.2] ):
    # extract {q,a,r}
    q, a, r = data['q'], data['r'], data['respect']
    # number of examples
    data_len = len(q)
    # splice indices
    lens = [ int(data_len*item) for item in ratio ]

    train = {
            'q' : q[:lens[0]],
            'a' : a[:lens[0]],
            'r' : r[:lens[0]]
            }

    test =  {
            'q' : q[lens[0]:lens[0]+lens[1]],
            'a' : a[lens[0]:lens[0]+lens[1]],
            'r' : r[lens[0]:lens[0]+lens[1]]
            }

    valid = {
            'q' : q[-lens[-1]:],
            'a' : a[-lens[-1]:],
            'r' : r[-lens[-1]:]
            }

    return train, test, valid


'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(d, batch_size):
    q, a, r = np.array(d['q']), np.array(d['a']), np.array(d['r'])
    while True:
        sample_idx = sample(list(np.arange(len(q))), batch_size)
        qi, ai, ri = q[sample_idx], a[sample_idx], r[sample_idx]
        qi_lens = (qi != 0).sum(1)
        ai_lens = (ai != 0).sum(1)
        yield qi.T[:qi_lens.max()].T, ai.T[:ai_lens.max()].T, ri


'''
 a generic decode function 
    inputs : sequence, lookup

'''
def decode(sequence, lookup, separator=' '): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])

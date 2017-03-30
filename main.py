import numpy as np

from datasets.friends import data
from datasets.friends import data_utils

import argparse
from model import conditioned_seq2seq
from vanilla import seq2seq


'''
    parse arguments

'''
def parse_args():
    parser = argparse.ArgumentParser(
            description='Train Model for Goal Oriented Dialog Task : bAbI(6)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--interact', action='store_true',
                        help='perform inference in an interactive session')
    group.add_argument('--ui', action='store_true',
                        help='interact through web app(flask); do not call this from cmd line (not implemented!)')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    parser.add_argument('--state_size', required=False, type=int, default=32,
                        help='Number of hidden units')
    parser.add_argument('--num_layers', required=False, type=int, default=2,
                        help='Num layers of stacked RNN')
    parser.add_argument('--batch_size', required=False, type=int, default=64,
                        help='you know what batch size means!')
    parser.add_argument('--eval_interval', required=False, type=int, default=10,
                        help='num iteration of training over train set')
    parser.add_argument('--log_file', required=False, type=str, default='log.txt',
                        help='enter the name of the log file')
    parser.add_argument('--save', required=False, action='store_false',
                        help='save checkpoints')
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--vanilla', action='store_true',
                        help='Simple sequence to sequence model')
    group1.add_argument('--conditioned', action='store_true',
                        help='conditioned seq2seq model')
    args = vars(parser.parse_args())
    return args


def train(model_fn, data_, metadata, args):
    # split into train/test/valid
    train, test, valid = data_utils.split_dataset(data_)
    # prepare train set generator
    #  
    batch_size = args['batch_size']  # replace with cmd line args
    trainset = data_utils.rand_batch_gen(train, batch_size=batch_size)
    validset = data_utils.rand_batch_gen(valid, batch_size=batch_size)

    ###
    # infer vocab size
    vocab_size = len(metadata['idx2w'])
    ext_context_size = metadata['respect_size']
    #
    # create a model
    if model_fn == conditioned_seq2seq:
        model = model_fn(state_size=32, vocab_size=vocab_size,
                                    num_layers=2, ext_context_size=ext_context_size,
                                    batch_size=batch_size)
    else:
        model = model_fn(state_size=32, vocab_size=vocab_size,
                                    num_layers=2,
                                    batch_size=batch_size)
    # train
    model.train(trainset, validset, n=len(train['q'])//(batch_size*2),
                valid_n=len(valid['q'])//(batch_size*2),
                epochs=100000, save=args['save'])


class InteractiveSession():

    def __init__(self, model_fn, metadata, args):
        # init interaction
        def init_interaction():
            # attach metadata to self
            self.metadata = metadata
            # infer vocab size, context size
            vocab_size = len(metadata['idx2w'])
            ext_context_size = metadata['respect_size']
            #
            # create a model
            if model_fn == conditioned_seq2seq:
                self.model = model_fn(state_size=args['state_size'],
                        vocab_size=vocab_size, num_layers=args['num_layers'],
                        ext_context_size=ext_context_size, batch_size=1)
            else:
                self.model = model_fn(state_size=args['state_size'],
                        vocab_size=vocab_size, num_layers=args['num_layers'],
                        batch_size=1)

            # restore last checkpoint (note_to_self :  i should stop writing redundant comments)
            self.model.restore_last_checkpoint()
        # call init
        init_interaction()

    def respond(self, query):
        #
        # [1] encode query<str>
        # [2] get predictions from model ([3] take argmax)
        # [4] decode response<array>
        enc_query = data_utils.encode(query, self.metadata['w2idx'])
        response = self.model.predict(enc_query)
        return ':: ' + data_utils.decode(response, self.metadata['idx2w'])


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    #
    # gather data
    data_, metadata = data.load_data(PATH='datasets/friends/')
    # get model type from args
    model_fn = conditioned_seq2seq if args['conditioned'] else seq2seq
    # train
    if args['train']:
        train(model_fn, data_, metadata, args)
    # interactive session
    elif args['interact']:
        isess = InteractiveSession(model_fn, metadata, args)
        while True:
            try:
                print(isess.respond(input('>> ')))
            except KeyboardInterrupt:
                break


''' 
    SKOL!
          oOOOOOo
         ,|    oO
        //|     |
        \\|     |
          `-----`
'''

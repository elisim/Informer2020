from informer_args_from_colab import informer_args
from exp.exp_informer import Exp_Informer as Exp
import numpy as np


def train():
    print("In train")
    args = informer_args

    # Change arguments
    args.train_epochs = 1
    args.itr = 1
    # args.data_path = 'ETTh1_one_row.csv'  # data file

    for ii in range(args.itr):  # TODO REMOVE LOOP
        # set experiments
        exp = Exp(args)

        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}' \
                  '_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
                                                                                    args.data,
                                                                                    args.features,
                                                                                    args.seq_len,
                                                                                    args.label_len,
                                                                                    args.pred_len,
                                                                                    args.d_model,
                                                                                    args.n_heads,
                                                                                    args.e_layers,
                                                                                    args.d_layers,
                                                                                    args.d_ff,
                                                                                    args.attn,
                                                                                    args.factor,
                                                                                    args.embed,
                                                                                    args.distil,
                                                                                    args.mix,
                                                                                    args.des,
                                                                                    ii)
        # train
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # # test
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting)

    print("Out train")
    return setting


if __name__ == '__main__':
    setting = train()
    print(setting)
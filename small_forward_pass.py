class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TimeSeriesTransformerConfig:
    """
    Based on
    https://huggingface.co/docs/transformers/model_doc/time_series_transformer
    """
    args = dotdict()

    args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]

    args.data = 'ETTh1'  # data
    args.root_path = './ETDataset/ETT-small/'  # root path of data file
    args.data_path = 'ETTh1.csv'  # data file
    args.features = 'M'  # forecasting task, options:[M, S, MS];

    "M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate"
    # I'm not understand what it means. Multivariate predict multivariate
    # I think it "input_size" in HF
    # I think: There are vars in the data depend in time, but the target is univariate.
    # Let's check the data sets.

    #
    # config = \
    #     {'activation': 'gelu',
    #  'attn': 'prob',
    #  'batch_size': 32,
    #  'c_out': 7,
    #  'checkpoints': './informer_checkpoints',
    #  'd_ff': 2048,
    #  'd_layers': 1,
    #  'd_model': 512,
    #  'data': 'ETTh1',
    #  'data_path': 'ETTh1.csv',
    #  'dec_in': 7,
    #  'des': 'exp',
    #  'detail_freq': 'h',
    #  'devices': '0,1,2,3',
    #  'distil': True,
    #  'dropout': 0.05,
    #  'e_layers': 2,
    #  'embed': 'timeF',
    #  'enc_in': 7,
    #  'factor': 5,
    #  'features': 'M',
    #  'freq': 'h',
    #  'gpu': 0,
    #  'itr': 1,
    #  'label_len': 48,
    #  'learning_rate': 0.0001,
    #  'loss': 'mse',
    #  'lradj': 'type1',
    #  'mix': True,
    #  'model': 'informer',
    #  'n_heads': 8,
    #  'num_workers': 0,
    #  'output_attention': False,
    #  'padding': 0,
    #  'patience': 3,
    #  'pred_len': 24,
    #  'root_path': './ETDataset/ETT-small/',
    #  'seq_len': 96,
    #  'target': 'OT',
    #  'train_epochs': 6,
    #  'use_amp': False,
    #  'use_gpu': False,
    #  'use_multi_gpu': False}


def get_config():
   # from https://huggingface.co/blog/time-series-transformers
   # config = TimeSeriesTransformerConfig(
   #   prediction_length=prediction_length,
   #   context_length=prediction_length * 3,  # context length
   #   lags_sequence=lags_sequence,
   #   num_time_features=len(time_features) + 1,  # we'll add 2 time features ("month of year" and "age", see further)
   #   num_static_categorical_features=1,  # we have a single static categorical feature, namely time series ID
   #   cardinality=[len(train_dataset)],  # it has 366 possible values
   #   embedding_dimension=[2],  # the model will learn an embedding of size 2 for each of the 366 possible values
   #   encoder_layers=4,
   #   decoder_layers=4,
   # )


def main():
    config = get_config()


if __name__ == '__main__':
    main()



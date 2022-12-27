class TimeSeriesTransformerConfig:
    """
    Based on
    https://huggingface.co/docs/transformers/model_doc/time_series_transformer
    """
    config = \
        {'activation': 'gelu',
     'attn': 'prob',
     'batch_size': 32,
     'c_out': 7,
     'checkpoints': './informer_checkpoints',
     'd_ff': 2048,
     'd_layers': 1,
     'd_model': 512,
     'data': 'ETTh1',
     'data_path': 'ETTh1.csv',
     'dec_in': 7,
     'des': 'exp',
     'detail_freq': 'h',
     'devices': '0,1,2,3',
     'distil': True,
     'dropout': 0.05,
     'e_layers': 2,
     'embed': 'timeF',
     'enc_in': 7,
     'factor': 5,
     'features': 'M',
     'freq': 'h',
     'gpu': 0,
     'itr': 1,
     'label_len': 48,
     'learning_rate': 0.0001,
     'loss': 'mse',
     'lradj': 'type1',
     'mix': True,
     'model': 'informer',
     'n_heads': 8,
     'num_workers': 0,
     'output_attention': False,
     'padding': 0,
     'patience': 3,
     'pred_len': 24,
     'root_path': './ETDataset/ETT-small/',
     'seq_len': 96,
     'target': 'OT',
     'train_epochs': 6,
     'use_amp': False,
     'use_gpu': False,
     'use_multi_gpu': False}


def get_config():
    pass


def main():
    config = get_config()


if __name__ == '__main__':
    main()


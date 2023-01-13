from informer_args_from_colab import informer_args
from exp.exp_informer import Exp_Informer as Exp


def predict():
    setting = "informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0"
    args = informer_args
    exp = Exp(args)
    preds = exp.predict(setting, True)
    print(preds)


if __name__ == '__main__':
    predict()
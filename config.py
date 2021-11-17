import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config_model_ver_2():
    return  {'vocab_size':10000,
                    'max_length_text':30,
                    'max_cate':14,
                    'max_length_cate':5,
                    'max_size':9,
                    'max_length_size':5,
                    'max_style':12,
                    'max_length_style':11,
                    'price_dim':3,
                    'img_dim':401,
                    'out_dim_embed':50,
                    'out_dim_lstm':100,
                    'out_dim_linear':500,
                    'raw_merge_feature_dim': 3072,     # 512*6
                    'device': device,
                    }
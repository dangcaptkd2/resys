import torch
from torch import nn


class Resys(nn.Module): 
    def __init__(self, X):
        super(Resys, self).__init__()

        self.embedding_text = nn.Embedding(X['vocab_size'], X['out_dim_embed'])  # shape [batch size, 30, 50]
        self.linear_text_0 = nn.Linear(X['max_length_text']*X['out_dim_embed'],1024)
        self.linear_text_1 = nn.Linear(1024,512)

        self.embedding_cate = nn.Embedding(X['max_cate'], X['out_dim_embed'])  # [batch size, 5] 
        self.linear_cate_0 = nn.Linear(X['max_length_cate']*X['out_dim_embed'],512)
        self.linear_cate_1 = nn.Linear(512,512)

        self.embedding_size = nn.Embedding(X['max_size'], X['out_dim_embed'])  # [batch size, 5] 
        self.linear_size_0 = nn.Linear(X['max_length_size']*X['out_dim_embed'],512)
        self.linear_size_1 = nn.Linear(512,512)


        self.embedding_style = nn.Embedding(X['max_style'], X['out_dim_embed'])  # [batch style, 11] 
        self.linear_style_0 = nn.Linear(X['max_length_style']*X['out_dim_embed'],512)
        self.linear_style_1 = nn.Linear(512,512)

        self.linear_price = nn.Linear(X['price_dim'], X['out_dim_linear']) # [batch size, 3]
        self.linear_price_0 = nn.Linear(X['out_dim_linear'],512)
        self.linear_price_1 = nn.Linear(512,512)

        self.linear_img = nn.Linear(X['img_dim'], X['out_dim_linear']) # [batch size, 401]
        self.linear_img_0 = nn.Linear(X['out_dim_linear'],512)
        self.linear_img_1 = nn.Linear(512,512)

        self.linear_feature_0 = nn.Linear(X['raw_merge_feature_dim'], 1024)
        self.linear_feature_1 = nn.Linear(1024, 512)
        self.linear_feature_2 = nn.Linear(512, 256)

        self.device = X['device']
        self.relu_layer = nn.ReLU()

    def forward(self, img, price, cate, text, size, style):

        text_fea = self.embedding_text(text)
        text_fea = text_fea.view(text_fea.shape[0], 1, -1)  #[batch size, 30, 50] -> [batch size, 1500]

        linear_text_0 = self.linear_text_0(text_fea)
        linear_text_0 = self.relu_layer(linear_text_0)

        linear_text_1 = self.linear_text_1(linear_text_0)
        linear_text_1 = self.relu_layer(linear_text_1)

        linear_text_1 = linear_text_1.squeeze(1)

        #######################

        cate_fea = self.embedding_cate(cate)
        cate_fea = cate_fea.view(cate_fea.shape[0], 1, -1)  #[batch size, 5, 50] -> [batch size, 250]
        
        linear_cate_0 = self.linear_cate_0(cate_fea)
        linear_cate_0 = self.relu_layer(linear_cate_0)

        linear_cate_1 = self.linear_cate_1(linear_cate_0)
        linear_cate_1 = self.relu_layer(linear_cate_1)

        linear_cate_1 = linear_cate_1.squeeze(1)

        ############

        size_fea = self.embedding_size(size)
        size_fea = size_fea.view(size_fea.shape[0], 1, -1)  #[batch size, 5, 50] -> [batch size, 250]
        
        linear_size_0 = self.linear_size_0(size_fea)
        linear_size_0 = self.relu_layer(linear_size_0)

        linear_size_1 = self.linear_size_1(linear_size_0)
        linear_size_1 = self.relu_layer(linear_size_1)

        linear_size_1 = linear_size_1.squeeze(1)

        #################

        style_fea = self.embedding_style(style)
        style_fea = style_fea.view(style_fea.shape[0], 1, -1)  #[batch style, 11, 50] -> [batch style, 550]
        
        linear_style_0 = self.linear_style_0(style_fea)
        linear_style_0 = self.relu_layer(linear_style_0)

        linear_style_1 = self.linear_style_1(linear_style_0)
        linear_style_1 = self.relu_layer(linear_style_1)

        linear_style_1 = linear_style_1.squeeze(1)

        ###################

        price_fea = self.linear_price(price)  # [batch size, 500]
        price_fea = self.relu_layer(price_fea)

        linear_price_0 = self.linear_price_0(price_fea)
        linear_price_0 = self.relu_layer(linear_price_0)

        linear_price_1 = self.linear_price_1(linear_price_0)
        linear_price_1 = self.relu_layer(linear_price_1)
        
        #####################

        img_fea = self.linear_img(img)  # [batch size, 500]
        img_fea = self.relu_layer(img_fea)

        linear_img_0 = self.linear_img_0(img_fea)
        linear_img_0 = self.relu_layer(linear_img_0)

        linear_img_1 = self.linear_img_1(linear_img_0)
        linear_img_1 = self.relu_layer(linear_img_1)

        merge_feature = torch.cat((linear_text_1, linear_cate_1, linear_price_1, linear_img_1, linear_size_1, linear_style_1), dim=1) # [batch_size, 3072] 
        
        feature_0 = self.linear_feature_0(merge_feature)
        feature_0 = self.relu_layer(feature_0)
        feature_1 = self.linear_feature_1(feature_0)
        feature_1 = self.relu_layer(feature_1)
        feature_2 = self.linear_feature_2(feature_1)

        return feature_2
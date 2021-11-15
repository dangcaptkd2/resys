import pickle
import json
import numpy as np
import torch
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset

class UserDataset(Dataset):
        def __init__(self):
                self.features = pickle.load(open('./data/features.pkl', "rb"))
                self.triple_data = json.load(open('./data/triple_data.json'))

                self.text_idx = 0
                self.cate_idx = 1
                self.img_idx = 2
                self.price_idx = 3
                self.media_idx = 4
                self.product_idx = 5
                self.size_idx = 6
                self.style_idx = 7

                texts = []
                for key, value in self.features.items():
                    texts.append(value[0])
                texts = np.array(texts)
                self.init_tokenizer(texts)


        def init_tokenizer(self, texts):
            # Max number of words in each item name.
            self.MAX_SEQUENCE_LENGTH = 30
            self.MAX_NB_WORDS = 10000

            self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS,lower=True)
            self.tokenizer.fit_on_texts(texts)

            print('Init successfully tokenizer!!!!')
        
        def tokenizer_text(self, text):
            X = self.tokenizer.texts_to_sequences(text)
            X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
            
            return X
        
        def get_img(self,anchors_id, poss_id, negs_id):
                img_anchors = []
                for i in range(len(anchors_id)):
                    anchor_img = self.features[anchors_id[i]][self.img_idx]
                    #anchor_img = anchor_img.reshape(1,401)
                    img_anchors.append(anchor_img)
                    
                pos_img = self.features[poss_id][self.img_idx]
                neg_img = self.features[negs_id][self.img_idx]

                return torch.tensor(img_anchors,dtype=torch.float), torch.tensor(pos_img,dtype=torch.float), \
                                torch.tensor(neg_img,dtype=torch.float)

        def get_cate(self, anchors_id, poss_id, negs_id):
                cate_anchors = []
                for i in range(len(anchors_id)):
                    anchor_cate = self.features[anchors_id[i]][self.cate_idx]
                    #anchor_cate = np.eye(13)[anchor_cate]
                    anchor_cate = np.pad([anchor_cate], (0,4), constant_values=13)
                    cate_anchors.append(anchor_cate)
                    
                pos_cate = self.features[poss_id][self.cate_idx]
                #pos_cate = np.eye(13)[pos_cate]
                pos_cate = np.pad([pos_cate], (0,4), constant_values=13)

                neg_cate = self.features[negs_id][self.cate_idx]
                #neg_cate = np.eye(13)[neg_cate]
                neg_cate = np.pad([neg_cate], (0,4), constant_values=13)


                return torch.tensor(cate_anchors,dtype=torch.int), torch.tensor(pos_cate,dtype=torch.int), \
                                torch.tensor(neg_cate,dtype=torch.int)

        def get_size(self, anchors_id, poss_id, negs_id):
                size_anchors = []
                for i in range(len(anchors_id)):
                    anchor_size = self.features[anchors_id[i]][self.size_idx]
                    #anchor_size = np.eye(8)[anchor_size]
                    anchor_size = np.pad([anchor_size], (0,4), constant_values=8)
                    size_anchors.append(anchor_size)
                    
                pos_size = self.features[poss_id][self.size_idx]
                #pos_size = np.eye(8)[pos_size]
                pos_size = np.pad([pos_size], (0,4), constant_values=8)

                neg_size = self.features[negs_id][self.size_idx]
                #neg_size = np.eye(8)[neg_size]
                neg_size = np.pad([neg_size], (0,4), constant_values=8)

                return torch.tensor(size_anchors,dtype=torch.int), torch.tensor(pos_size,dtype=torch.int), \
                                torch.tensor(neg_size,dtype=torch.int)

        def get_style(self, anchors_id, poss_id, negs_id):
                style_anchors = []
                for i in range(len(anchors_id)):
                    anchor_style = self.features[anchors_id[i]][self.style_idx]
                    num_pad = 11-len(anchor_style)
                    anchor_style = np.pad(anchor_style, (0,num_pad), constant_values=11)
                    style_anchors.append(anchor_style)
                    
                pos_style = self.features[poss_id][self.style_idx]
                num_pad = 11-len(pos_style)
                pos_style = np.pad(pos_style, (0,num_pad), constant_values=11)

                neg_style = self.features[negs_id][self.style_idx]
                num_pad = 11-len(neg_style)
                neg_style = np.pad(neg_style, (0,num_pad), constant_values=11)

                return torch.tensor(style_anchors,dtype=torch.int), torch.tensor(pos_style,dtype=torch.int), \
                                torch.tensor(neg_style,dtype=torch.int)

        def get_text(self, anchors_id, poss_id, negs_id):
                text_anchors = []
                for i in range(len(anchors_id)):
                    raw_text_anchor = self.features[anchors_id[i]][self.text_idx]
                    anchor_text = self.tokenizer_text([raw_text_anchor])    
                    text_anchors.append(anchor_text[0])
                    
                raw_text_pos = self.features[poss_id][self.text_idx]
                pos_text = self.tokenizer_text([raw_text_pos])
                pos_text = pos_text[0]

                raw_text_neg = self.features[negs_id][self.text_idx]
                neg_text = self.tokenizer_text([raw_text_neg])
                neg_text = neg_text[0]

                return torch.tensor(text_anchors,dtype=torch.int), torch.tensor(pos_text,dtype=torch.int), \
                                torch.tensor(neg_text,dtype=torch.int)

        def get_pmp_score(self, anchors_id, poss_id, negs_id):
                pmp_score_anchors = []
                for i in range(len(anchors_id)):
                    price = self.features[anchors_id[i]][self.price_idx]
                    media_score = self.features[anchors_id[i]][self.media_idx]
                    product_score = self.features[anchors_id[i]][self.product_idx]

                    anchor_pmp_score = np.array([price, media_score, product_score])
                
                    pmp_score_anchors.append(anchor_pmp_score)
                    

                price = self.features[poss_id][self.price_idx]
                media_score = self.features[poss_id][self.media_idx]
                product_score = self.features[poss_id][self.product_idx]
                pos_pmp_score = np.array([price, media_score, product_score])

                price = self.features[negs_id][self.price_idx]
                media_score = self.features[negs_id][self.media_idx]
                product_score = self.features[negs_id][self.product_idx]
                neg_pmp_score = np.array([price, media_score, product_score])

                return torch.tensor(pmp_score_anchors,dtype=torch.float), torch.tensor(pos_pmp_score,dtype=torch.float), \
                                torch.tensor(neg_pmp_score,dtype=torch.float)

        def  __getitem__(self, id):
                data_point = self.triple_data[id]

                anchors_id = data_point[0]
                poss_id = data_point[1]   
                negs_id = data_point[2]


                img_anchor, img_pos, img_neg = self.get_img(anchors_id, poss_id, negs_id)
                price_anchor, price_pos, price_neg = self.get_pmp_score(anchors_id, poss_id, negs_id)
                cate_anchor, cate_pos, cate_neg = self.get_cate(anchors_id, poss_id, negs_id)
                text_anchor, text_pos, text_neg = self.get_text(anchors_id, poss_id, negs_id)
                size_anchor, size_pos, size_neg = self.get_size(anchors_id, poss_id, negs_id)
                style_anchor, style_pos, style_neg = self.get_style(anchors_id, poss_id, negs_id)

                anchor = {'img': img_anchor, 'price': price_anchor, 'cate': cate_anchor, 'text': text_anchor, 'size': size_anchor, 'style': style_anchor}
                
                pos = {'img': img_pos, 'price': price_pos, 'cate': cate_pos, 'text': text_pos, 'size': size_pos, 'style': style_pos}

                neg = {'img': img_neg, 'price': price_neg, 'cate': cate_neg, 'text': text_neg, 'size': size_neg, 'style': style_neg}
                        

        
                return anchor, pos, neg

        def __len__(self):
            return len(self.triple_data)


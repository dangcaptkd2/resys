import torch.optim as optim
import torch
import wandb
from torch.utils.data import DataLoader

import numpy as np

from model import Resys
from dataloader import UserDataset
from loss import TripletCosineLoss

from tqdm import tqdm

wandb.login()
torch.manual_seed(2)

def train(num_epoches, model, train_loader):
    for epoch in tqdm(range(num_epoches), desc="Epochs"):
        running_loss = []
        running_dap = []
        running_dan = []
        for step, (anchors, positive, negative) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            optimizer.zero_grad()
        
            #get positive features
            img_pos = positive['img'].to(device)
            text_pos = positive['text'].to(device)
            price_pos = positive['price'].to(device)
            cate_pos = positive['cate'].to(device)
            size_pos = positive['size'].to(device)
            style_pos = positive['style'].to(device)
            positive_out = model(img_pos, price_pos, cate_pos, text_pos, size_pos, style_pos)

            #get negative features
            img_neg = negative['img'].to(device)
            text_neg = negative['text'].to(device)
            price_neg = negative['price'].to(device)
            cate_neg = negative['cate'].to(device)
            size_neg = negative['size'].to(device)
            style_neg = negative['style'].to(device)
            negative_out = model(img_neg, price_neg, cate_neg, text_neg, size_neg, style_neg)

            #get list of anchor features
            list_anchor = []
            for i in range(anchors['img'].shape[1]):
                img_anchor = anchors['img'][:,i,:].to(device)
                text_anchor = anchors['text'][:,i,:].to(device)
                price_anchor = anchors['price'][:,i,:].to(device)
                cate_anchor = anchors['cate'][:,i,:].to(device)
                size_anchor = anchors['size'][:,i,:].to(device)
                style_anchor = anchors['style'][:,i,:].to(device)
                anchor_out = model(img_anchor, price_anchor, cate_anchor, text_anchor, size_anchor, style_anchor)
                list_anchor.append(anchor_out)

            anchor_out_tensor = torch.stack(list_anchor, dim=0)
            
            # compute triplet loss
            anchor_out_tensor = torch.permute(anchor_out_tensor, (1, 0, 2))
            positive_out = positive_out.unsqueeze(1)
            negative_out = negative_out.unsqueeze(1)

            loss, dap, dan = criterion(anchor_out_tensor, positive_out, negative_out)

            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.cpu().detach().numpy())
            running_dap.append(dap.cpu().detach().numpy())
            running_dan.append(dan.cpu().detach().numpy())

        #save and show loss
        cur_loss = np.mean(running_loss)
        cur_dap = np.mean(running_dap)
        cur_dan = np.mean(running_dan)
        tracking_loss.append(cur_loss)
        if cur_loss<=min(tracking_loss):
            torch.save(model.state_dict(), "./checkpoint/best_checkpoint_fulldata.pt")
        torch.save(model.state_dict(), "./checkpoint/checkpoint_fulldata.pt")
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, num_epoches, cur_loss))
        wandb.log({
            "Train loss": cur_loss,
            "Distance(anchor, positive)": cur_dap,
            "Distance(anchor, negative)": cur_dan,
            })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_input_model = {'vocab_size':10000,
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
model = Resys(init_input_model)
model.to(device)
model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = TripletCosineLoss()

train_ds = UserDataset()
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

wandb.init(entity="dangcaptkd", project="recys-model", notes="ver_2", tags="ssh_3")
model = model.float()
model.train()
tracking_loss = []
num_epoches = 150

train(num_epoches, model, train_loader)

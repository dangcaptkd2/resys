import torch
from tqdm import tqdm
import numpy as np
import faiss
import wandb

from model import Resys
from config import get_config_model_ver_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint):
    config = get_config_model_ver_2()
    model = Resys(config)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model.eval()
    return model 

def get_embedding(model, val_loader):
    d = {}
    for post in tqdm(val_loader):
        img = post['img'].to(device)
        text = post['text'].to(device)
        price = post['price'].to(device)
        cate = post['cate'].to(device)
        size = post['size'].to(device)
        style = post['style'].to(device)

        post_out = model(img, price, cate, text, size, style)
        d[int(post['post_id'])] = post_out[0].cpu().detach().numpy().tolist()
    return d

def evaluate(model, val_loader, user_ids, user_10_post, user_pos_lst, thres=0.98):

    d = get_embedding(model, val_loader)

    list_ = list(map(int, list(d.keys())))
    arr = []
    for k in list_:
        arr.append(d[k])
    arr=np.array(arr)

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(256))
    arr = np.float32(arr)
    faiss.normalize_L2(arr)
    index.add_with_ids(arr, np.array(list_))

    total_acc = []
    for user_id in tqdm(user_ids):
        positives = user_pos_lst[user_id]
        acc_post = []
        ids_relate = []
        distance_relate = []
        index_pos = []
        for post_id_anchor in user_10_post[user_id]:
            idx = list_.index(post_id_anchor)
            key = arr[idx].reshape(1,256)
            k=1000
            D, I = index.search(key, k)  # search
            result_ids = list(I[0])
            dis = D[0]
            ids_relate.extend(result_ids)
            distance_relate.extend(dis)
        try:
            index_pos = [ids_relate.index(i) for i in positives]
        except:
            pass
        if index_pos:
            for i in index_pos:
                if distance_relate[i] > thres:
                    acc_post.append(ids_relate[i])
        

        acc_post = list(set(acc_post))
        accuracy_1_user = len(acc_post) / len(positives)
        total_acc.append(accuracy_1_user)

    return np.mean(total_acc)

def train(num_epoches, model, train_loader, criterion, optimizer, tracking_loss):
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


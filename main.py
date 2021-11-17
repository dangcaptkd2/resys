import torch.optim as optim
import torch
import wandb
from torch.utils.data import DataLoader

import numpy as np

from model import Resys
from dataloader import UserDataset
from loss import TripletCosineLoss
from config import get_config_model_ver_2
from utils import train

from tqdm import tqdm

wandb.login()
torch.manual_seed(2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = get_config_model_ver_2()
model = Resys(config)
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

train(num_epoches, model, train_loader, criterion, optimizer, tracking_loss)

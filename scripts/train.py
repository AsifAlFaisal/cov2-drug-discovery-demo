#%% imports
from custom_data import Cov2Data
from model import Cov2GEN
from utils import *
import torch

#%% load model, data and split data
model = Cov2GEN(hidden_dim=64, output_dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
dataset = Cov2Data(root="../data/", filename='cov2_inhibitors.csv')
train_loader, test_loader = data_splitter(dataset, fraction=0.8, batch_size=4)

#%% training
train(train_loader, model, optimizer, criterion, num_epochs=400, print_output=True)

# %% testing
model.load_state_dict(torch.load('../saved_model/Cov2GEN.pth'))
test(test_loader, model, criterion)

# %%

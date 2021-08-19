from custom_data import Cov2Data
from model import Cov2GEN
from utils import *
import torch

model = Cov2GEN(hidden_dim=64, output_dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
dataset = Cov2Data(root="../data/", filename='cov2_inhibitors.csv')
train_loader, test_loader = data_splitter(dataset, batch_size=4)

train(train_loader, model, optimizer, criterion, num_epochs=10, print_output=False)


## Some Utility functions for training

from torch_geometric.data import DataLoader
import torch
from tqdm import tqdm
import time

def data_splitter(data, fraction=0.8, batch_size=2):
    data = data.shuffle()
    train_size = int(len(data)*fraction)
    train_set, test_set = data[:train_size], data[train_size:]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2)
    return train_loader, test_loader

def train(loader, model, optimizer, criterion, num_epochs, print_output):
    for epoch in range(num_epochs):
        correct = 0
        batch_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
        for idx, data in pbar:
            time.sleep(0.05) # slowing down progress bar by 50 millisecond
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(torch.squeeze(out), data.y.float())
            pred = torch.tensor([1 if i>= 0.5 else 0 for i in (out.squeeze())])
            correct += (pred==data.y).sum()
            batch_loss += loss.item()
            loss.backward()  
            optimizer.step()

            pbar.set_description(f"Epoch: {epoch}/{num_epochs}")
            pbar.set_postfix({'step loss': loss.item()})
        train_acc = correct/len(loader.dataset)
        train_loss = batch_loss/len(loader.dataset)
        if print_output:
            print(f"EPOCH: {epoch+1}, Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .4f}")
## Some Utility functions for training

from torch_geometric.data import DataLoader
import torch
from tqdm import tqdm
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def data_splitter(data, fraction=0.8, batch_size=2):
    data = data.shuffle()
    train_size = int(len(data)*fraction)
    train_set, test_set = data[:train_size], data[train_size:]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader

def train(loader, model, optimizer, criterion, num_epochs, print_output):
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        batch_loss = 0
        best_loss = 1000.0
        pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
        for idx, data in pbar:
            #time.sleep(0.01) # slowing down progress bar by 10 millisecond
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
            if (epoch+1)%100==0:
                print(f"\nAfter EPOCH: {epoch+1}, Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), '../saved_model/Cov2GEN.pth')
    print("Best Model updated and stored.")


def test(loader, model, criterion):
    model.eval()
    correct = 0
    batch_loss = 0
    all_preds = []
    all_truths = []
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(torch.squeeze(out), data.y.float())
        pred = torch.tensor([1 if i>= 0.5 else 0 for i in (out.squeeze())])
        all_preds.extend(pred.detach().cpu().numpy())
        all_truths.extend(data.y.detach().cpu().numpy())
        correct += (pred==data.y).sum()
        batch_loss += loss.item()
    test_loss = batch_loss/len(loader.dataset)
    acc = accuracy_score(all_truths, all_preds)
    precision = precision_score(all_truths, all_preds)
    recall = recall_score(all_truths, all_preds)
    cfm = confusion_matrix(all_truths, all_preds)
    f1s = f1_score(all_truths, all_preds)
    _, ax = plt.subplots(figsize=(6,6)) 
    fig = sns.heatmap(cfm, annot=True, ax=ax)
    fig.figure.savefig("../saved_model/output_images/confusion_matrix_test.png")
    print(f"Test Results:\nLoss: {test_loss: .4f}, Accuracy: {acc: .4f}, \
            \nPrecision: {precision: .4f}, Recall: {recall: .4f}, F1-Score: {f1s: .4f} \nConfusion Matrix: {cfm}")
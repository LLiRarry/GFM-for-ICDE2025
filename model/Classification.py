import os
import time
import glob
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from models import *
from layers import *
from dataset import GraphClassificationDataset

dataset = GraphClassificationDataset(name='ffmpeg_6ACFG_min3_max200')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GMS2(number_class=dataset.number_features,device=device)
# # 99.34
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
epochs = 1000



def train():
    print('\nModel training.\n')
    start = time.time()
    val_loss_values = []
    best_epoch = 0
    min_loss = 1e10
    best_model_path = ''

    model.train()
    for epoch in range(epochs):
        main_index = 0
        loss_sum = 0
        batches = dataset.create_batches(dataset.training_funcs, dataset.collate)
        for index, batch_pair in enumerate(batches):
            optimizer.zero_grad()
            data = dataset.transform(batch_pair)
            data['target'].to(device)
            prediction = model(data)
            loss = F.binary_cross_entropy(prediction, data['target'], reduction='sum')
            loss.backward()
            optimizer.step()
            main_index += len(batch_pair[2])
            loss_sum += loss.item()
        loss = loss_sum / main_index

        if epoch + 1 < 950:
            end = time.time()
            print('Epoch: {:05d}, loss_train: {:.6f}, time: {:.6f}s'.format(epoch + 1, loss, end - start))
        else:
            val_loss, aucscore = validate(dataset, dataset.validation_funcs)
            end = time.time()
            print('Epoch: {:05d}, loss_train: {:.6f}, loss_val: {:.6f}, AUC: {:.6f}, time: {:.6f}s'.format(epoch + 1, loss, val_loss, aucscore, end - start))
            val_loss_values.append(val_loss)
            if val_loss < min_loss:
                min_loss = val_loss
                best_epoch = epoch
                if best_model_path:
                    os.remove(best_model_path)
                best_model_path = '{}.pth'.format(epoch)
                torch.save(model.state_dict(), best_model_path)

    print('Optimization Finished! Total time elapsed: {:.6f}s'.format(time.time() - start))
    print('Best model saved as: {}'.format(best_model_path))
    return best_epoch

def validate(datasets, funcs):
    model.eval()
    main_index = 0
    loss_sum = 0
    with torch.no_grad():
        pred = []
        gt = []
        batches = datasets.create_batches(funcs, datasets.collate)
        for index, batch_pair in enumerate(batches):
            data = datasets.transform(batch_pair)
            data['target'].to(device)
            prediction = model(data)
            loss = F.binary_cross_entropy(prediction, data['target'], reduction='sum')
            main_index = main_index + len(batch_pair[2])
            loss_sum = loss_sum + loss.item()

            batch_gt = batch_pair[2]
            batch_pred = list(prediction.detach().cpu().numpy())

            pred = pred + batch_pred
            gt = gt + batch_gt

        loss = loss_sum / main_index
        gt = np.array(gt, dtype=np.float32)
        pred = np.array(pred, dtype=np.float32)
        score = roc_auc_score(gt, pred)

        return loss, score


if __name__ == "__main__":
    best_model = train()
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    print('\nModel evaluation.')
    test_loss, test_auc = validate(dataset, dataset.testing_funcs)
    print('Test set results, loss = {:.6f}, AUC = {:.6f}'.format(test_loss, test_auc))

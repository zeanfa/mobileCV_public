#!/bin/env python3

import pandas as pd
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from torchvision import models
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Список имён файлов
images_special_car = listdir('../Dataset/Special_car/')
image_rest = listdir('../Dataset/Rest/')

# Переносим их в датафрейм
data_special_car = pd.DataFrame(images_special_car, columns=['src'])
data_rest = pd.DataFrame(image_rest, columns=['src'])

# Признаки
data_special_car['label'] = 1
data_rest['label'] = 0

# Объядиняем
data = pd.concat([data_special_car, data_rest], ignore_index=True)

X = []
y = []
for i in tqdm(range(len(data))):
    image = data.loc[i, 'src']
    if (data.loc[i, 'label'] == 1):
        image = cv2.imread('../Dataset/Special_car/' + image, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread('../Dataset/Rest/ ' + image, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(200, 200), interpolation=cv2.INTER_AREA)
    X.append(image)
    y.append(data.loc[i, 'label'])

# Нормализация
X_torch = np.array(X)
X_torch = X_torch.astype('float32')
X_torch = X_torch / 255.0
X_torch = X_torch.reshape(-1, 3, 200, 200)
y_torch = np.array(y).reshape(-1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

X_train, X_valid, y_train, y_valid = train_test_split(X_torch, y_torch, random_state=42)

# Формирование пакетов для тензора
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train)
X_valid_t = torch.from_numpy(X_valid).float()
y_valid_t = torch.from_numpy(y_valid)
train_dataset = TensorDataset(X_train_t, y_train_t)
valid_dataset = TensorDataset(X_valid_t, y_valid_t)
train_dataloader = DataLoader(train_dataset, batch_size=31)
valid_dataloader = DataLoader(valid_dataset, batch_size=32)

model = models.densenet169()
model.classifier = torch.nn.Linear(1664, 2)
model.to(device)

criterion = torch.nn.CrossEntropyLoss() # Ошибки
optimizer = torch.optim.Adam(model.parameters()) # Градиент ошибок
loaders = {"train": train_dataloader, "valid": valid_dataloader}

max_epochs = 500
best_model = model
last_loss = np.Inf
epoch_erly_stopping = 0
flag = False
col_not_best = 0
accyracy_best = 0 # Лучшая точность
accuracy_history = {"train": [], "valid": []}
loss_history = {"train": [], "valid": []}
train_losses = []
valid_losses = []
start_time = datetime.now()

for epoch in range(max_epochs + 1):
    for k, dataloader in loaders.items():
        epoch_correct = 0
        epoch_all = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if k == "train":
                model.train()
                optimizer.zero_grad()
                outp = model(x_batch)
            else:
                model.eval()
                with torch.no_grad():
                    outp = model(x_batch)

            preds = outp.argmax(-1)
            correct = (preds == y_batch).sum()
            all = preds.size(0)
            epoch_correct += correct.item()
            epoch_all += all
            loss = criterion(outp, y_batch) # Градиеты
            # loss = criterion(outp.squeeze(), y_batch.float())
            if k == "train":
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            else:
                valid_losses.append(loss.item())

        if k == "valid":
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            loss_history["train"].append(train_loss)
            loss_history["valid"].append(valid_loss)
            print(f"[{epoch:>3}/{max_epochs:>3}] loss train: {train_loss:.5f} | loss valid: {valid_loss:.5f}")
            if round(last_loss, 5) > round(valid_loss, 5):
                col_not_best = 0
                best_model = model
                epoch_erly_stopping = epoch
                torch.save(best_model, "checkpoint_2.pt")
                print(f"Validation loss decreased ({last_loss:.5f} --> {valid_loss:.5f}).  Saving model ...")
                last_loss = valid_loss
            else:
                if col_not_best + 1 >= 20:
                    print("Early stopping!")
                    accuracy_history[k].append(epoch_correct / epoch_all)
                    flag = True
                    break
                else:
                    col_not_best += 1
                    print(f"EarlyStopping counter: {col_not_best} out of 20")
        accuracy_history[k].append(epoch_correct / epoch_all)
    if flag:
        break

torch.save(best_model, "densenet169.pth")
print(f'Program execution time: {datetime.now() - start_time}')

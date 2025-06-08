import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric import nn, data
import pytorch_lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import shutil
import math
import sys

from typing import List, Tuple, Union

from CustomDatasets import TrafficDataset
from CustomModels import GATLSTMModel, GCNLSTMModel, STGATModel

# # number of possible 5 minutes in a days. Formula: 24 (hours) * 60 (minutes/hour) / 5 (minutes) = 288
# POSSIBLE_SLOT = (24 * 60) // 5
# config = {
#     "F": 12, # Number of past time steps to consider
#     "H": 3, # Number of future time steps to predict
#     "N_DAYS": 44,
#     "N_DAY_SLOT": POSSIBLE_SLOT,
#     "BATCH_SIZE": 16, # Reduced from 50
#     "LR": 2e-4,
#     "WEIGHT_DECAY" : 5e-4
# }

# config["N_SLOT"] = config["N_DAY_SLOT"] - (config["H"] + config["F"]) + 1
# dataset = TrafficDataset(config, root="metr-la-dataset")

def split_dataset(
    dataset: TrafficDataset,
    possible_slot: int,
    split_days: tuple,
):
    n_train_day, n_test_day, _ = split_days
    i = int(n_train_day * possible_slot)
    j = int(n_test_day * possible_slot)
    train_dataset = dataset[:i]
    test_dataset = dataset[i : i + j]
    val_dataset = dataset[i + j :]

    return train_dataset, test_dataset, val_dataset

# train, test, val = split_dataset(
#     dataset=dataset,
#     possible_slot=config["N_SLOT"],
#     split_days=(34, 5, 5) # we have totally 44 days in the dataset, we use 34 days for train, 5 days for test and 5 days for validation
# )

# train_loader = tg.loader.DataLoader(train, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=True)
# test_loader = tg.loader.DataLoader(test, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)
# val_loader = tg.loader.DataLoader(val, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)

# # test the dataset
# print(f"Number of training samples: {len(train)}")
# print(f"Number of testing samples: {len(test)}")
# print(f"Number of validation samples: {len(val)}")
# print(f"Number of features: {dataset.data.x.shape[1]}")
# print(f"Number of nodes: {dataset.n_node}")
# print(f"Mean: {dataset.mean}")
# print(f"Std: {dataset.std}")

# # test the dataloader
# for batch in train_loader:
#     print(f"Batch size: {batch.batch_size}")
#     print(f"Number of nodes in batch: {batch.num_nodes}")
#     print(f"Number of edges in batch: {batch.edge_index.shape[1]}")
#     print(f"Batch x shape: {batch.x.shape}")
#     print(f"Batch y shape: {batch.y.shape}")
#     break  # just test the first batch
# # test the model
# timer = L.callbacks.Timer()
# early_stopping = L.callbacks.EarlyStopping(
#     monitor="val_loss",
#     patience=10,
#     mode="min")
# callbacks = [timer, early_stopping]

# # GCN LSTM model
# """
# class GCNLSTMModel(L.LightningModule):
#     def __init__(self,
#                     in_channel: int,
#                     gcn_hidden_channel: int,
#                     n_nodes: int,
#                     drop_out: float,
#                     lstm_dim: list[int],
#                     prediction_time_step: int,
#                     lr: float,
#                     weight_decay: float)
# """
# model = GCNLSTMModel(
#     in_channel=dataset.data.x.shape[1],
#     gcn_hidden_channel=64,
#     n_nodes=dataset.n_node,
#     drop_out=0.5,
#     lstm_dim=[64, 32],
#     prediction_time_step=config["H"],
#     lr=config["LR"],
#     weight_decay=config["WEIGHT_DECAY"]
# )

# gcn_trainer = L.Trainer(
#     accelerator="cuda" if torch.cuda.is_available() else "cpu",
#     devices=1 if torch.cuda.is_available() else None,
#     callbacks=callbacks,
#     precision=16 if torch.cuda.is_available() else 32,
#     max_epochs=30,
#     default_root_dir="./logs"
# )
# gcn_trainer.fit(model, train_loader, val_loader)
# gcn_trainer.test(model, test_loader)
# print(f"Train time: {timer.time_elapsed('train'):.3f}s")
# print(f"Validate time: {timer.time_elapsed('validate'):.3f}s")
# print(f"Test time: {timer.time_elapsed('test'):.3f}s")

# # GAT LSTM model
# """
# class GATLSTMModel(L.LightningModule):
#     def __init__(self,
#                     in_channel: int,
#                     gat_out_channel: int,
#                     n_nodes: int,
#                     att_heads: int,
#                     concat_gat: bool,
#                     drop_out: float,
#                     lstm_dim: list[int],
#                     prediction_time_step: int,
#                     lr: float,
#                     weight_decay: float)
# """
# gat_model = GATLSTMModel(
#     in_channel=dataset.data.x.shape[1],
#     gat_out_channel=32, # Optionally reduce GAT output channels
#     n_nodes=dataset.n_node,
#     att_heads=4, # Optionally reduce attention heads
#     concat_gat=True,
#     drop_out=0.5,
#     lstm_dim=[32, 16], # Optionally reduce LSTM dimensions
#     prediction_time_step=config["H"],
#     lr=config["LR"],
#     weight_decay=config["WEIGHT_DECAY"]
# )
# gat_trainer = L.Trainer(
#     accelerator="cuda" if torch.cuda.is_available() else "cpu",
#     devices=1 if torch.cuda.is_available() else None,
#     callbacks=callbacks,
#     precision=16 if torch.cuda.is_available() else 32,
#     max_epochs=30,
#     default_root_dir="./logs"
# )
# gat_trainer.fit(gat_model, train_loader, val_loader)
# gat_trainer.test(gat_model, test_loader)
# print(f"Train time: {timer.time_elapsed('train'):.3f}s")
# print(f"Validate time: {timer.time_elapsed('validate'):.3f}s")
# print(f"Test time: {timer.time_elapsed('test'):.3f}s")
# # plot

# plt.figure(figsize=(10, 5))
# plt.plot(range(gat_model.current_epoch), gat_model.history["loss"])
# plt.plot(range(gat_model.current_epoch), gat_model.history["val_loss"])
# plt.legend(["Train Loss", "Validation Loss"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss")
# plt.grid()
# plt.savefig("loss_plot.png")
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(range(gat_model.current_epoch), gat_model.history["train_MAE"])
# plt.plot(range(gat_model.current_epoch), gat_model.history["val_MAE"])
# plt.legend(["Train MAE", "Validation MAE"])
# plt.xlabel("Epoch")
# plt.ylabel("MAE")
# plt.title("Training and Validation MAE")
# plt.grid()
# plt.savefig("MAE_plot.png")
# plt.show()



# New interface with os and sys
"""
python main.py <model_name: gcn/gat/stgat> <horizon_number: int> <batch_size: int> 
"""
os.system("rm -rf metr-la-dataset/processed")
# config:
# number of possible 5 minutes in a days. Formula: 24 (hours) * 60 (minutes/hour) / 5 (minutes) = 288
POSSIBLE_SLOT = (24 * 60) // 5
config = {
    "F": 12, # Number of past time steps to consider
    "H": int(sys.argv[2]), # Number of future time steps to predict, taken from command line argument
    "N_DAYS": 44,
    "N_DAY_SLOT": POSSIBLE_SLOT,
    "BATCH_SIZE": int(sys.argv[3]), # Batch size taken from command line argument
    "LR": 2e-4,
    "WEIGHT_DECAY" : 5e-4
}
# Load dataset
if sys.argv[1] == "gcn":
    config["N_SLOT"] = config["N_DAY_SLOT"] - (config["H"] + config["F"]) + 1
    dataset = TrafficDataset(config, root="metr-la-dataset", gat_version=False)
elif sys.argv[1] == "gat" or sys.argv[1] == "stgat":
    config["N_SLOT"] = config["N_DAY_SLOT"] - (config["H"] + config["F"]) + 1
    dataset = TrafficDataset(config, root="metr-la-dataset", gat_version=True)
else:
    raise ValueError("Model name must be 'gcn', 'gat' or 'stgat'.")

# because we use inmemory dataset, we need to clean the memory and reload back the dataset
import gc
gc.collect()  # collect garbage
torch.cuda.empty_cache()  # clear GPU memory if using CUDA

# Load dataset after cleaning memory
if sys.argv[1] == "gcn":
    config["N_SLOT"] = config["N_DAY_SLOT"] - (config["H"] + config["F"]) + 1
    dataset = TrafficDataset(config, root="metr-la-dataset", gat_version=False)
elif sys.argv[1] == "gat" or sys.argv[1] == "stgat":
    config["N_SLOT"] = config["N_DAY_SLOT"] - (config["H"] + config["F"]) + 1
    dataset = TrafficDataset(config, root="metr-la-dataset", gat_version=True)
else:
    raise ValueError("Model name must be 'gcn', 'gat' or 'stgat'.")

# Split dataset
train, test, val = split_dataset(
    dataset=dataset,
    possible_slot=config["N_SLOT"],
    split_days=(30, 9, 5) # we have totally 44 days in the dataset, we use 30 days for train, 9 days for test and 5 days for validation
)

train_loader = tg.loader.DataLoader(train, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=True)
test_loader = tg.loader.DataLoader(test, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)
val_loader = tg.loader.DataLoader(val, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)

# test the dataset
print(f"Number of training samples: {len(train)}")
print(f"Number of testing samples: {len(test)}")
print(f"Number of validation samples: {len(val)}")
print(f"Number of features: {dataset.data.x.shape[1]}")
print(f"Number of nodes: {dataset.n_node}")
print(f"Mean: {dataset.mean}")
print(f"Std: {dataset.std}")

# test the dataloader
for batch in train_loader:
    print(f"Batch size: {batch.batch_size}")
    print(f"Number of nodes in batch: {batch.num_nodes}")
    print(f"Number of edges in batch: {batch.edge_index.shape[1]}")
    print(f"Batch x shape: {batch.x.shape}")
    print(f"Batch y shape: {batch.y.shape}")
    break  # just test the first batch
# test the model
timer = L.callbacks.Timer()
early_stopping = L.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min")
callbacks = [timer, early_stopping]

if sys.argv[1] == "gcn":
    # GCN LSTM model
    output_dir = Path(f"./logs/gcn_logs_{config['H']}_{config['BATCH_SIZE']}")
    model = GCNLSTMModel(
        in_channel=config["F"],
        gcn_hidden_channel=config["F"], # Optionally reduce GCN hidden channels
        n_nodes=dataset.n_node,
        drop_out=0.2,
        lstm_dim=[32, 128],
        prediction_time_step=config["H"],
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
        batch_size=config["BATCH_SIZE"] # Pass batch size to the model
    )
    trainer = L.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        precision=16 if torch.cuda.is_available() else 32,
        max_epochs=30,
        default_root_dir=output_dir
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    print(f"Train time: {timer.time_elapsed('train'):.3f}s")
    print(f"Validate time: {timer.time_elapsed('validate'):.3f}s")
    print(f"Test time: {timer.time_elapsed('test'):.3f}s")
elif sys.argv[1] == "gat":
    # GAT LSTM model
    output_dir = Path(f"./logs/gat_logs_{config['H']}_{config['BATCH_SIZE']}")
    model = GATLSTMModel(
        in_channel=config["F"],
        gat_out_channel=config["F"], # Optionally reduce GAT output channels
        n_nodes=dataset.n_node,
        att_heads=8, # Optionally reduce attention heads
        drop_out=0.2,
        lstm_dim=[32, 128], # Optionally reduce LSTM dimensions
        prediction_time_step=config["H"],
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
        batch_size=config["BATCH_SIZE"], # Pass batch size to the model
        concat_gat=True # Use concatenation for GAT output
    )
    trainer = L.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        precision=16 if torch.cuda.is_available() else 32,
        max_epochs=30,
        default_root_dir=output_dir
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    print(f"Train time: {timer.time_elapsed('train'):.3f}s")
    print(f"Validate time: {timer.time_elapsed('validate'):.3f}s")
    print(f"Test time: {timer.time_elapsed('test'):.3f}s")
elif sys.argv[1] == "stgat":
    # STGAT model
    output_dir = Path(f"./logs/stgat_logs_{config['H']}_{config['BATCH_SIZE']}")
    model = STGATModel(
        in_channel=config["F"],
        out_channel=config["F"], # Optionally reduce GAT output channels
        n_nodes=dataset.n_node,
        att_head_nodes=8,
        drop_out=0.2,
        lstm_dim=[32, 128], # Optionally reduce LSTM dimensions
        prediction_time_step=config["H"],
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
        batch_size=config["BATCH_SIZE"] # Pass batch size to the model
    )
    trainer = L.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        precision=16 if torch.cuda.is_available() else 32,
        max_epochs=30,
        default_root_dir=output_dir
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    print(f"Train time: {timer.time_elapsed('train'):.3f}s")
    print(f"Validate time: {timer.time_elapsed('validate'):.3f}s")
    print(f"Test time: {timer.time_elapsed('test'):.3f}s")

# Plotting MSE and MAE, RMSE
plt.figure(figsize=(10, 5))
plt.plot(range(model.current_epoch), model.history["loss"], label="Train Loss")
plt.plot(range(model.current_epoch), model.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss MSE")
plt.grid()
#save to the logs directory
plt.savefig(os.path.join(output_dir, f"loss_plot_{sys.argv[1]}.png"))
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(model.current_epoch), model.history["train_MAE"], label="Train MAE")
plt.plot(range(model.current_epoch), model.history["val_MAE"], label="Validation MAE")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training and Validation MAE")
plt.grid()
# save to the logs directory
plt.savefig(os.path.join(output_dir, f"MAE_plot_{sys.argv[1]}.png"))
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(model.current_epoch), model.history["train_RMSE"], label="Train RMSE")
plt.plot(range(model.current_epoch), model.history["val_RMSE"], label="Validation RMSE")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training and Validation RMSE")
plt.grid()
# save to the logs directory
plt.savefig(os.path.join(output_dir, f"RMSE_plot_{sys.argv[1]}.png"))
# plt.show()
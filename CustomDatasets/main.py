import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric import nn, data
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import shutil
import math

from typing import List, Tuple, Union

def z_score(data: torch.Tensor, mean: float, std: float):
    return (data - mean) / std

def reverse(data: torch.Tensor, mean: float, std: float):
    return (data * std) + mean

# for GAT
class TrafficDataset(data.InMemoryDataset):
    def __init__(
        self,
        config: dict,
        root: str,
        gat_version: bool = True,
        transform=None,
        pre_transform=None,
    ):
        self.config = config
        self.gat_version = gat_version
        super().__init__(root, transform, pre_transform)
        (
            self.data,
            self.slices,
            self.n_node,
            self.mean,
            self.std,
        ) = torch.load(self.processed_paths[0], weights_only=False) # Explicitly set weights_only=False

    # return the path of the file contains data which is processed
    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return ["./data.pt"]

    # The path to the file contains data
    @property
    def raw_file_names(self) -> str | List[str] | Tuple:
        return [
            os.path.join(self.raw_dir, "METR-LA.h5"),
            os.path.join(self.raw_dir, "adj_METR-LA.pkl"),
        ]

    # download the raw dataset file
    def download(self):
        V_dest = os.path.join(self.raw_dir, "METR-LA.h5")
        W_dest = os.path.join(self.raw_dir, "adj_METR-LA.pkl")
        shutil.copyfile(os.path.join(self.root, "METR-LA.h5"), V_dest)
        shutil.copyfile(os.path.join(self.root, "adj_METR-LA.pkl"), W_dest)

    def process(self):
        df = pd.read_hdf(self.raw_file_names[0], "df")
        *_, weight_df = pd.read_pickle(self.raw_file_names[1])
        W = self._distance_to_weight(torch.from_numpy(weight_df), gat_version=self.gat_version)
        data_ = torch.from_numpy(df.values)
        mean = torch.mean(data_)
        std = torch.std(data_)
        data_ = z_score(data_, mean, std)
        _, num_nodes = data_.shape
        edge_index = torch.zeros((2, num_nodes**2), dtype=torch.long)
        edge_label = torch.zeros((num_nodes**2, 2))
        num_edges = 0
        # extract edge list from adjacency matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if W[i, j] != 0:
                    edge_index[0, num_edges] = i
                    edge_index[1, num_edges] = j
                    edge_label[num_edges] = W[i, j]
                    num_edges += 1

        # resize edge list from number_nodes^2
        edge_index = edge_index.resize_((2, num_edges))
        edge_label = edge_label.resize_(num_edges, 1)
        sequences = self._speed2vec(
            edge_index,
            edge_label,
            num_nodes,
            self.config["N_DAYS"],
            self.config["N_SLOT"],
            data_,
            self.config["F"],
            self.config["H"],
        )
        data_, slices = self.collate(sequences)

        torch.save(
            (data_, slices, num_nodes, mean, std),
            self.processed_paths[0],
        )

    def _distance_to_weight(
        self,
        W: torch.tensor,
        sigma2: float = 0.1,
        epsilon: float = 0.5,
        gat_version: bool = False,
    ):
        num_nodes = W.shape[0]
        BASE_KM = 10_000.0
        W = W / BASE_KM
        W2 = W * W
        W_mask = torch.ones([num_nodes, num_nodes]) - torch.eye(num_nodes)
        W = (
            torch.exp(-W2 / sigma2)
            * (torch.exp(-W2 / sigma2) >= epsilon)
            * W_mask
        )

        if gat_version:
            W[W > 0] = 1
            W += torch.eye(num_nodes)

        return W

    def _speed2vec(
        self,
        edge_index: torch.tensor,
        edge_label: torch.tensor,
        num_nodes: int,
        n_days: int,
        n_slot: int,
        data_: torch.tensor,
        F: int,
        H: int,
    ):
        window_length = F + H
        sequences = []
        for i in range(n_days):
            for j in range(n_slot):
                G = data.Data()
                G.__num_nodes__ = num_nodes
                G.edge_index = edge_index
                G.edge_label = edge_label

                start = i * F + j
                end = start + window_length
                # transpose
                full_windows = data_[start:end:].T
                G.x = full_windows[:, 0:F]
                G.y = full_windows[:, F::]
                sequences.append(G)

        return sequences
    
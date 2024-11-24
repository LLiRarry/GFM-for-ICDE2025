
from models import *
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau
from torch_geometric.data import DataLoader, Batch
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree, sort_edge_index
from torch_geometric.datasets import GEDDataset
from utils import calculate_ranking_correlation, calculate_prec_at_k
from torch_geometric.loader import DataLoader
import torch

class GNNTrainer(object):

    def __init__(self):
        self.pe = []
        self.pe_test = []
        self.model = GMS(number_class=89)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device, "d")
        self.model.to(self.device)  # Move model to the specified device
        self.epochs = 20000

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """

        self.training_graphs = GEDDataset(
            "../IMDBMulti", "IMDBMulti", train=True
        )
        self.testing_graphs = GEDDataset(
            "../IMDBMulti", "IMDBMulti", train=False
        )

        self.nged_matrix = self.training_graphs.norm_ged  # 700x700,一共700个图，当然是训练集上的

        self.real_data_size = self.nged_matrix.size(0)
        if self.training_graphs[0].x is None:
            # print("None")
            max_degree = 0
            for g in (
                    self.training_graphs
                    + self.testing_graphs
            ):
                if g.edge_index.size(1) > 0:
                    max_degree = max(
                        max_degree, int(degree(g.edge_index[0]).max().item())
                    )
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree



    def create_batches(self):

        source_loader = DataLoader(
            self.training_graphs,
            shuffle=True,
            batch_size=32,
        )
        target_loader = DataLoader(
            self.training_graphs,
            shuffle=True,
            batch_size=32,
        )

        return list(zip(source_loader, target_loader))

    def transform(self, data):
        new_data = dict()
        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        normalized_ged = self.nged_matrix[
            data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()
        ].tolist()
        new_data["target"] = (
            torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        )

        return new_data

    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data.
        """
        self.optimizer.zero_grad()
        # data = self.transform(data)  # transform
        data = {k: v.to(self.device) for k, v in self.transform(data).items()}
        target = data["target"]
        prediction = self.model(data)
        loss = F.mse_loss(prediction, target, reduction="sum")
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self):
        # self.model.load_state_dict(torch.load("model_pt/model.pt"))
        initial_lr = 0.001
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=initial_lr,
            weight_decay=0.0001
        )

        # Define a custom lambda function for learning rate decay
        def lr_lambda(epoch):
            if epoch >= 3000:
                return 0.95 ** ((epoch - 3000) // 1000)
            else:
                return 1

        scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.model.train()
        epochs = trange(self.epochs, leave=True, desc="Epoch")
        loss_list = []
        best_mse = float('inf')  # Initialize best MSE as infinity
        for epoch in epochs:
            self.optimizer.step()  # Update optimizer for current epoch's learning rate
            scheduler.step()  # Update learning rate scheduler
            batches = self.create_batches()  # Assuming a function to create data batches
            main_index = 0
            loss_sum = 0
            for index, batch_pair in tqdm(
                    enumerate(batches), total=len(batches), desc="Batches", leave=False
            ):
                loss_score = self.process_batch(batch_pair)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
            loss = loss_sum / main_index
            current_lr = self.optimizer.param_groups[0]['lr']
            epochs.set_description(f"Epoch {epoch} (Loss={round(loss, 5)}, LR={current_lr})")
            # epochs.set_description("Epoch %d (Loss=%g)" % (epoch, round(loss, 5)))
            torch.save(self.model.state_dict(), f"model_imdb.pt")
            loss_list.append(loss)

        return loss_list

    def score(self):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()

        scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))

        rho_list = []
        tau_list = []
        prec_at_10_list = []
        prec_at_20_list = []

        t = tqdm(total=len(self.testing_graphs) * len(self.training_graphs))

        for i, g in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([g] * len(self.training_graphs))
            target_batch = Batch.from_data_list(self.training_graphs)

            # data = self.transform((source_batch, target_batch))
            data = {k: v.to(self.device) for k, v in self.transform((source_batch, target_batch)).items()}
            target = data["target"]
            ground_truth[i] = target
            prediction = self.model(data)
            prediction_mat[i] = prediction.detach().numpy()

            scores[i] = (
                F.mse_loss(prediction, target, reduction="none").detach().numpy()
            )

            rho_list.append(
                calculate_ranking_correlation(
                    spearmanr, prediction_mat[i], ground_truth[i]
                )
            )
            tau_list.append(
                calculate_ranking_correlation(
                    kendalltau, prediction_mat[i], ground_truth[i]
                )
            )
            prec_at_10_list.append(
                calculate_prec_at_k(10, prediction_mat[i], ground_truth[i])
            )
            prec_at_20_list.append(
                calculate_prec_at_k(20, prediction_mat[i], ground_truth[i])
            )

            t.update(len(self.training_graphs))

        self.rho = np.mean(rho_list).item()
        self.tau = np.mean(tau_list).item()
        self.prec_at_10 = np.mean(prec_at_10_list).item()
        self.prec_at_20 = np.mean(prec_at_20_list).item()
        self.model_error = np.mean(scores).item()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error * 1000, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")
        # torch.save(self.model.state_dict(), "model_pt/model.pt")


if __name__ == "__main__":
    trainer = GNNTrainer()
    trainer.process_dataset()
    trainer.fit()
    trainer.score()
    trainer.print_evaluation()

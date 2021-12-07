from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from sklearn import metrics
from torch_geometric_temporal.signal import temporal_signal_split


class Trainer:
    def __init__(self, model, dataset, args, testset=None):
        self.device = args.device
        self.model = model.to(self.device)
        self.train_set, self.val_set = temporal_signal_split(
            dataset, train_ratio=1 - args.val_size
        )
        self.test_set = testset
        self.epochs = args.num_epochs
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.best_model_weights = self.model.state_dict()
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.verbose = args.verbose

    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            train_loss = []
            val_loss = []
            for i, data in enumerate(self.train_set):
                loss = self._train_step(self.model, data)
                train_loss.append(loss)

            for i, data in enumerate(self.val_set):
                loss = self._train_step(self.model, data)
                val_loss.append(loss)
            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)
            if self.verbose:
                if epoch % 20 == 0 or epoch == self.epochs:
                    print(
                        f"""
                        epoch {epoch}:
                            train loss: {train_loss},
                            val loss: {val_loss}
                    """
                    )
            if self.best_val_loss > val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_weights = self.model.state_dict()
        self.model.load_state_dict(self.best_model_weights)
        print(
            f"""
            best model loss is:
                val loss: {self.best_val_loss} @ epoch: {self.best_epoch}
            """
        )
        self._benchmark()
        return self.model

    def _train_step(self, model, data):
        self.optimizer.zero_grad()
        logits, target = self._shared_step(model, data)
        loss = self.criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _val_step(self, model, data):
        with torch.no_grad():
            logits, target = self._shared_step(model, data)
            loss = self.criterion(logits, target)
            return loss.item()

    def _shared_step(self, model, data):
        data.x = data.x.to(self.device)
        data.edge_index = data.edge_index.to(self.device)
        data.edge_attr = data.edge_attr.to(self.device)
        target = (data.y > 0).long().to(self.device)
        logits = model(data)
        return logits, target

    def _benchmark(self):
        train_preds = []
        train_trues = []
        val_preds = []
        val_trues = []
        for i, data in enumerate(self.train_set):
            logits, target = self._shared_step(self.model, data)
            pred = logits.argmax(-1).cpu().numpy()
            target = target.cpu().numpy()
            train_preds.append(pred)
            train_trues.append(target)
        for i, data in enumerate(self.val_set):
            logits, target = self._shared_step(self.model, data)
            pred = logits.argmax(-1).cpu().numpy()
            target = target.cpu().numpy()
            val_preds.append(pred)
            val_trues.append(target)
        train_preds = np.hstack(train_preds)
        train_trues = np.hstack(train_trues)
        val_preds = np.hstack(val_preds)
        val_trues = np.hstack(val_trues)
        print("==================validation set performance==================")
        print(f"ROC AUC score {metrics.roc_auc_score(val_trues, val_preds)}")
        print(metrics.classification_report(val_trues, val_preds))
        if self.test_set:
            print("==================test set performance==================")
            test_preds = []
            test_trues = []
            for i, data in enumerate(self.test_set):
                logits, target = self._shared_step(self.model, data)
                pred = logits.argmax(-1).cpu().numpy()
                target = target.cpu().numpy()
                test_preds.append(pred)
                test_trues.append(target)
            test_preds = np.hstack(test_preds)
            test_trues = np.hstack(test_trues)
            print(f"ROC AUC score {metrics.roc_auc_score(test_trues, test_preds)}")
            print(metrics.classification_report(test_trues, test_preds))

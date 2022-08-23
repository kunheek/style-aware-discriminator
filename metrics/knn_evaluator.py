import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
from model.augmentation import SimpleTransform
from .base_evaluator import BaseEvaluator


class KNNEvaluator(BaseEvaluator):

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument(
            "--nb-knn", type=int, default=5,
            help="Number of NN to use."
        )
        return parser

    def __init__(self, nb_knn, temperature, **kwargs):
        self.nb_knn = nb_knn
        self.temperature = temperature
        super().__init__(**kwargs)

    def prepare_evaluation(self):
        print("Preparing k-NN evaluation ...")
        if "lsun" in self.train_dataset or "lsun" in self.eval_dataset:
            return self._is_available

        transform = SimpleTransform(self.image_size)
        self.trainset = data.ImageFolder(self.train_dataset, transform, True)
        self.valset = data.ImageFolder(self.eval_dataset, transform, True)

        if not hasattr(self.trainset, "classes") \
            or len(self.trainset.classes) < 2:
            return self._is_available
        self.num_classes = len(self.trainset.classes)
        self._is_available = True
        print("Done!")
        return self._is_available

    @torch.no_grad()
    def evaluate(self, model, dataset=None, batch_size=50, step=None):
        print("Evaluating k-NN accuracy.")
        # Set non-shuffle train loader + extract class-wise images
        loader_kwargs = {"num_workers": 4, "pin_memory": True}
        train_loader = DataLoader(self.trainset, batch_size, **loader_kwargs)
        val_loader = DataLoader(self.valset, batch_size, **loader_kwargs)

        train_features, train_labels = extract_features(model, train_loader)
        test_features, test_labels = extract_features(model, val_loader)
        del train_loader
        del val_loader

        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

        # dump features
        filename = os.path.join(self.run_dir, "features/{}_{:06d}.pt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(train_features.cpu(), filename.format("trainfeat", step))
        torch.save(train_labels.cpu(), filename.format("trainlabels", step))
        torch.save(test_features.cpu(), filename.format("testfeat", step))
        torch.save(test_labels.cpu(), filename.format("testlabels", step))

        top1, top5 = knn_classifier(
            train_features, train_labels, test_features, test_labels,
            k=self.nb_knn, T=self.temperature, num_classes=self.num_classes,
        )
        return {"top1": top1, "top5": top5}


@torch.no_grad()
def extract_features(model, data_loader):
    features, labels = [], []
    for x, y in data_loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        feats = model.D_ema(x, command="encode").clone()
        features.append(feats)
        labels.append(y)

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


# Code adopted from 
# https://github.com/facebookresearch/dino/blob/main/eval_knn.py
@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    # train_features: |train_ds| x feature_dim
    # train_features: |train_ds|
    # train_features: |test_ds| x feature_dim
    # train_features: |test_ds|
    # k: how many neighbors
    # T: temperature
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        if num_classes > 5:
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5

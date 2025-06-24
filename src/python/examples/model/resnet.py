import os
import subprocess
import fsspec

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50, resnet101, resnet152
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class ResNet18(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.01, momentum=0.9, weight_decay=5e-4, weights=None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(512, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


class ResNet50(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.01, momentum=0.9, weight_decay=5e-4, weights=None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet50(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


class ResNet101(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.01, momentum=0.9, weight_decay=5e-4, weights=None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet101(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


class ResNet152(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.01, momentum=0.9, weight_decay=5e-4, weights=None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet152(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


def export_inference_model(
        model: pl.LightningModule, out_path: str, tmpdir: str
) -> None:
    """
    export_inference_model uses TorchScript JIT to serialize the
    TinyImageNetModel into a standalone file that can be used during inference.
    TorchServe can also handle interpreted models with just the model.py file if
    your model can't be JITed.
    """

    print("Exporting inference model")
    jit_path = os.path.join(tmpdir, "model_jit.pt")
    jitted = torch.jit.script(model)
    print(f"Saving JIT model to {jit_path}")
    torch.jit.save(jitted, jit_path)

    model_name = str(model.__class__.__name__)

    mar_path = os.path.join(tmpdir, f"{model_name}.mar")
    print(f"Creating model archive at {mar_path}")
    subprocess.run(
        [
            "torch-model-archiver",
            "--model-name",
            model_name,
            "--handler",
            "torchx/examples/apps/lightning/handler/handler.py",
            "--version",
            "1",
            "--serialized-file",
            jit_path,
            "--export-path",
            tmpdir,
        ],
        check=True,
    )

    remote_path = os.path.join(out_path, f"{model_name}.mar")
    print(f"Uploading to {remote_path}")
    fs, _, rpaths = fsspec.get_fs_token_paths(remote_path)
    assert len(rpaths) == 1, "must have single path"
    fs.put(mar_path, rpaths[0])

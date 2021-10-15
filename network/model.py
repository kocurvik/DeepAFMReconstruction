import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from network.unet import ResUnet
import torch.nn.functional as F


class ResUnetModel(LightningModule):
    def __init__(self):
        super().__init__()


    # def train_dataloader(self):
    #     train_dataset = Dataset(args.path, 'train', preload=not args.no_preload)
    #     return DataLoader(train_dataset, batch_size=self.batch_size | self.hparams.batch_size)

    def forward(self, x):
        return self.model(x)[:, 0, ...]

    def training_step(self, batch, batch_idx):
        input = batch['input']
        gt = batch['gt']

        pred = self.model(input)
        loss = F.mse_loss(pred[:, 0, ...], gt)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch['input']
        gt = batch['gt']

        pred = self.model(input)
        loss = F.mse_loss(pred[:, 0, ...], gt)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        input = batch['input']
        gt = batch['gt']

        pred = self.model(input)
        loss = F.mse_loss(pred[:, 0, ...], gt)

        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
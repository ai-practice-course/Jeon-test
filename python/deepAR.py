from tqdm.notebook import tqdm as tqdm

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import MinMaxScaler

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


logger = TensorBoardLogger("../lightning_log", name="deepAR")

from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE, MultivariateNormalDistributionLoss, NormalDistributionLoss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-checkpoint')
parser.add_argument('-hidden', type = int)
parser.add_argument('-nlayer', type = int)

if __name__ == '__main__':
    args = parser.parse_args()

    # Load dataset
    training = TimeSeriesDataSet.load('../training_timeseriesdataset.tsd')
    validation = TimeSeriesDataSet.load('../validation_timeseriesdataset.tsd')
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-3, 
        patience=10, 
        verbose=True, 
        mode="min"
    )
    
    lr_logger = LearningRateMonitor()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = '../lightning_log/deepAR/checkpoints',
        save_top_k = 10,
        monitor = 'val_loss',
        mode = 'min',
        filename = '{epoch:02d}-{val_loss:.2f}'
    )
    
    batch_size = 64
    train_dataloader = training.to_dataloader(train = True, batch_size = batch_size, num_workers = 0)
    val_dataloader = validation.to_dataloader(train = False, batch_size = batch_size, num_workers = 0)
    
    if args.hidden:
        hidden = args.hidden
    else:
        hidden = 30
        
    if args.nlayer:
        nlayer = args.nlayer
    else:
        nlayer = 2
        
        
    ## Network
    net = DeepAR.from_dataset(
        training, 
        learning_rate = 1e-3, 
        hidden_size = hidden, 
        rnn_layers = nlayer, 
        dropout = 0,
        loss = MultivariateNormalDistributionLoss()
    )
    
    if args.checkpoint:
        net = net.load_from_checkpoint(f'/home/kyle/tmp_lecture/jeon-test/lightning_log/deepAR/{args.checkpoint}')
        
    ## Trainer
    trainer = pl.Trainer( 
        max_epochs = 1000,
        gpus = 1,
        weights_summary = 'top', ##
        gradient_clip_val = .2,
        callbacks = [lr_logger, ], # checkpoint_callback, early_stop_callback
        limit_train_batches = 30,
        enable_checkpointing = True,
        logger = logger,
        auto_lr_find = True
    )
    
    # best_model_path = trainer.checkpoint_callback.best_model_path
    # print('best model path = ', best_model_path)
    # best_model = net.load_from_checkpoint(best_model_path)
    
    
    trainer.fit(
        net,
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader,
    )
    
    predictions = net.predict(val_dataloader)
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    print(f'sMAPE = {SMAPE()(actuals, predictions)}')
    
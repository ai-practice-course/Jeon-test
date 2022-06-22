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
from pytorch_forecasting.metrics import SMAPE, MultivariateNormalDistributionLoss




if __name__ == '__main__':
    
    # Load dataset
    training = TimeSeriesDataSet.load('../training_timeseriesdataset.tsd')
    validation = TimeSeriesDataSet.load('../validation_timeseriesdataset.tsd')
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=10, 
        verbose=True, 
        mode="min"
    )
    
    lr_logger = LearningRateMonitor()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = '../lightning_log/DeepAR',
        save_top_k = 10,
        monitor = 'val_loss',
        mode = 'min',
        filename = 'deepar-{epoch:02d}-{val_loss:.2f}'
    )
    
    batch_size = 128
    train_dataloader = training.to_dataloader(train = True, batch_size = batch_size, num_workers = 0)
    val_dataloader = validation.to_dataloader(train = False, batch_size = batch_size, num_workers = 0)
    
    ## Network
    net = DeepAR.from_dataset(
        training, 
        learning_rate = 1e-3, 
        hidden_size = 128, 
        rnn_layers = 4, 
    loss = MultivariateNormalDistributionLoss()
    )
    
    ## Trainer
    trainer = pl.Trainer(
        max_epochs = 100,
        gpus = 1,
        weights_summary = 'top', ##
        gradient_clip_val = .01,
        callbacks = [lr_logger, early_stop_callback, checkpoint_callback],
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
    
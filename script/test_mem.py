import gc
import pdb
from collections import Counter

import pytorch_lightning as pl
import torch
import torchio as tio
from torch.utils.data import DataLoader
import resource
import time

n_subjects = 16
max_length = 40
samples_per_volume = 5
num_workers = 8
patch_size = 128
batch_size = 2

sampler = tio.data.UniformSampler(patch_size)
subject = tio.datasets.Colin27()
dataset = tio.SubjectsDataset(n_subjects * [subject])
queue = tio.Queue(dataset, max_length, samples_per_volume, sampler, num_workers)


class DummyDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(queue, batch_size=batch_size)


class DummyModule(pl.LightningModule):
    def configure_optimizers(self):
        pass

    def training_step(self, *args, **kwargs):
        #pdb.set_trace()  # Use inspect_mem() here.
        time.sleep(0.1)
        main_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
        child_memory = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss /1000
        print(f'Peak: {main_memory + child_memory} MB') 



def inspect_mem():
    tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.ndim == 4:
            tensors.append(obj.size())

    for item, count in Counter(tensors).items():
        print(item, count, sep=": ")


trainer = pl.Trainer(max_epochs=10, num_sanity_val_steps=0)
trainer.fit(DummyModule(), datamodule=DummyDataModule())


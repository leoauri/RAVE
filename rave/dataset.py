from typing import Iterable, Tuple, List
import numpy as np
import contextlib
import os, yaml
import gin.torch
import acids_dataset as ad
from scipy.signal import lfilter


@gin.configurable(module="data")
def split_dataset(dataset, 
                  partition: str | None = None, 
                  percent: float | None = None, 
                  features: str | List[str] | None = None, 
                  balance_cardinality: bool = True, 
                  max_residual: int | None = None,
                  training_name: str | None = None):
    if partition is not None:
        assert percent is None, "either percent or partition keyword must be given."
        partdict = dataset.load_partition(partition)
    elif (training_name is not None) and (dataset.has_partition(training_name)): 
        partdict = dataset.load_partition(training_name)
    else: 
        split1 = max((percent * len(dataset)) // 100, 1)
        split2 = len(dataset) - split1
        if max_residual is not None:
            split2 = min(max_residual, split2)
            split1 = len(dataset) - split2
        partdict = dataset.split(partitions={'train': split1 / len(dataset), 'test': split2 / len(dataset)}, 
                                 features=features, 
                                 balance_cardinality=balance_cardinality,
                                 write=training_name)
    return partdict['train'], partdict['test']


@gin.configurable
def get_dataset(db_path,
                sr,
                n_signal,
                derivative: bool = False,
                normalize: bool = False,
                rand_pitch: Tuple[float, float] | None = False,
                augmentations: Iterable[ad.transforms.Transform] | None = None, 
                n_channels: int = 1):

    metadata = ad.get_metadata_from_path(db_path)
    sr_dataset = metadata.get('sr', 44100)

    transform_list = []
    # resample
    if sr_dataset != sr:
        transform_list.append(ad.transforms.AudioResample(sr_dataset, sr))
    # stretch or crop
    if rand_pitch:
        transform_list.append(ad.transforms.Stretch(n_signal, rand_pitch, sr=sr))
    else:
        transform_list.append(ad.transforms.Crop(n_signal, sr=sr))
        
    transform_list.extend([
        ad.transforms.Dequantize(16, sr=sr),
        ad.transforms.PhaseMangle(sr=sr),
    ])

    if normalize:
        transform_list.append(ad.transforms.Normalize(sr=sr))

    if derivative:
        transform_list.append(ad.transforms.Derivator(sr=sr)[0])

    if augmentations:
        transform_list.extend(augmentations)

    return ad.datasets.AudioDataset(
        db_path,
        transforms=transform_list,
        channels=n_channels
    )


def get_channels_from_dataset(db_path):
    with open(os.path.join(db_path, 'metadata.yaml'), 'r') as metadata:
        metadata = yaml.safe_load(metadata)
    return metadata.get('channels')

def get_training_channels(db_path, target_channels):
    dataset_channels = get_channels_from_dataset(db_path)
    if dataset_channels is not None:
        if target_channels > dataset_channels:
            raise RuntimeError('[Error] Requested number of channels is %s, but dataset has %s channels')%(FLAGS.channels, dataset_channels)
    n_channels = target_channels or dataset_channels
    if n_channels is None:
        print('[Warning] channels not found in dataset, taking 1 by default')
        n_channels = 1
    return n_channels


# Utilitary for GIN recording of augmentations


_augmentations = []

@gin.configurable(module="transforms")
def parse_transform(transform):
    global _augmentations
    _augmentations.append(transform)

def get_augmentations():
    return _augmentations

def get_derivator_integrator(sr: int):
    alpha = 1 / (1 + 1 / sr * 2 * np.pi * 10)
    derivator = ([.5, -.5], [1])
    integrator = ([alpha**2, -alpha**2], [1, -2 * alpha, alpha**2])
    return lambda x: lfilter(*derivator, x), lambda x: lfilter(*integrator, x)


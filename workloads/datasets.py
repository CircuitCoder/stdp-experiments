"""Native-resolution image loading and deterministic class/sample selection."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zd3.io import load_mnist
from workloads.provenance import sha256


@dataclass(frozen=True)
class Images:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    classes: tuple[int, ...]
    image_shape: tuple[int, ...]
    manifest: dict


def most_frequent(labels, count):
    classes, frequencies = np.unique(labels, return_counts=True)
    # Frequency is the primary key; numeric label breaks ties.
    order = np.lexsort((classes, -frequencies))
    if count < 1 or count > len(classes):
        raise ValueError('Invalid number of retained classes')
    return tuple(sorted(int(label) for label in classes[order[:count]]))


def read_cifar(path):
    records = np.fromfile(path, dtype=np.uint8)
    if records.size == 0 or records.size % 3073:
        raise ValueError(f'Invalid CIFAR binary record size: {path}')
    records = records.reshape(-1, 3073)
    if np.any(records[:, 0] > 9):
        raise ValueError('Invalid CIFAR label')
    # Official binary layout: label, 1024 red, 1024 green, 1024 blue.
    return records[:, 1:].copy(), records[:, 0].copy()


def load_images(dataset, root, top_classes=None):
    root = Path(root)
    if dataset == 'fashion-mnist':
        train, test = load_mnist(root, 'train'), load_mnist(root, 'test')
        x, y, tx, ty = train.images, train.labels, test.images, test.labels
        shape = (28, 28)
        files = [root / name for name in ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                                          't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')]
    elif dataset == 'cifar10':
        files = [root / f'data_batch_{i}.bin' for i in range(1, 6)] + [root / 'test_batch.bin']
        training = [read_cifar(path) for path in files[:5]]
        x = np.concatenate([pair[0] for pair in training])
        y = np.concatenate([pair[1] for pair in training])
        tx, ty = read_cifar(files[-1])
        shape = (3, 32, 32)
    else:
        raise ValueError(f'Unsupported image dataset {dataset}')
    histogram = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    classes = most_frequent(y, top_classes or len(histogram))
    train_keep, test_keep = np.isin(y, classes), np.isin(ty, classes)
    record = {'dataset': dataset, 'native_shape': shape, 'pixel_layout': 'channel-major' if len(shape) == 3 else 'row-major',
              'training_histogram_before_filter': histogram, 'selected_labels': classes,
              'selection': 'training frequency descending, label ascending on ties; filter only',
              'train_count': int(train_keep.sum()), 'test_count': int(test_keep.sum()),
              'file_sha256': {p.name: sha256(p) for p in files}, 'root': str(root.resolve())}
    return Images(x[train_keep], y[train_keep], tx[test_keep], ty[test_keep], classes, shape, record)


def stratified_indices(labels, classes, per_class, seed):
    rng = np.random.RandomState(seed)
    chosen = []
    for label in classes:
        indices = np.flatnonzero(labels == label)
        if len(indices) < per_class:
            raise ValueError(f'Insufficient samples for class {label}')
        chosen.extend(rng.permutation(indices)[:per_class].tolist())
    return rng.permutation(chosen).astype(np.int64)


def score_activity(marking, marking_labels, testing, testing_labels, classes):
    means = np.stack([marking[marking_labels == label].mean(axis=0) for label in classes])
    assignments = np.asarray(classes)[np.argmax(means, axis=0)]
    assignments[means.max(axis=0) == 0] = -1
    scores = np.zeros((len(testing), len(classes)))
    for index, label in enumerate(classes):
        members = assignments == label
        if members.any():
            scores[:, index] = testing[:, members].mean(axis=1)
    # Class order is ascending; ties choose the first retained class.
    predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
    confusion = [[int(np.sum((testing_labels == actual) & (predictions == predicted)))
                  for predicted in classes] for actual in classes]
    return {'accuracy_percent': float(np.mean(predictions == testing_labels) * 100),
            'assignment_counts': {int(label): int(np.sum(assignments == label)) for label in classes},
            'unassigned_neurons': int(np.sum(assignments < 0)),
            'silent_test_samples': int(np.sum(testing.sum(axis=1) == 0)),
            'confusion_matrix': confusion, 'class_order': classes}

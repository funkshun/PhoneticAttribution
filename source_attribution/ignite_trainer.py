import argparse
import os
import datetime

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, _functional_tensor

from ignite.engine import (
    Engine,
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Precision, Recall, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine

from microdetector import PhonemicMicrodetector
import microdetector
from microphoneme_dataset import (
    MicrophonemeDataset,
    InterpolatedMelSpectrogram,
    InterpolatedMelFbank,
    CropInterpMelSpectrogram,
)


def binary_transform(output):
    y_pred, y = output
    y_pred, y = torch.round(y_pred).to(torch.int), y.to(torch.int)
    return y_pred, y


def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp += (2 * sum(np.logical_and(y_true[i], y_pred[i]))) / (
            sum(y_true[i]) + sum(y_pred[i])
        )
    return temp / y_true.shape[0]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--phoneme", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--train_class_size", type=int, default=None)
parser.add_argument("--valid_class_size", type=int, default=None)

parser.add_argument("--sr", type=int, default=16000)
parser.add_argument("--n_fft", type=int, default=512)
parser.add_argument("--hop_length", type=int, default=128)

parser.add_argument("--target_spec_length", type=int, default=128)

parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.001)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = InterpolatedMelSpectrogram(
#     target_len=args.target_spec_length,
#     sr=args.sr,
#     n_fft=args.n_fft,
#     hop_length=args.hop_length,
# )

# transform = CropInterpMelSpectrogram(
#     target_len=args.target_spec_length,
#     sr=args.sr,
#     n_fft=args.n_fft,
#     hop_length=args.hop_length,
# )
#
transform = InterpolatedMelFbank(
    target_len=args.target_spec_length,
    sr=args.sr,
    n_fft=args.n_fft,
    hop_length=args.hop_length,
)

trainloader = DataLoader(
    MicrophonemeDataset(
        os.path.join(args.dataset, "train"),
        args.phoneme,
        args.metadata,
        args.train_class_size,
        transform=transform,
        sr=args.sr,
    ),
    shuffle=True,
    batch_size=args.batch,
)

validloader = DataLoader(
    MicrophonemeDataset(
        os.path.join(args.dataset, "validation"),
        args.phoneme,
        args.metadata,
        args.valid_class_size,
        transform=transform,
        sr=args.sr,
    ),
    shuffle=False,
    batch_size=args.batch,
)


model = PhonemicMicrodetector(
    args.phoneme,
    (128, args.target_spec_length),
    len(trainloader.dataset.labels),
    device=device,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# criterion = nn.MultiLabelSoftMarginLoss()
criterion = nn.BCELoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device)

val_metrics = {
    "loss": Loss(criterion),
    "precision": Precision(
        average=True, is_multilabel=True, output_transform=binary_transform
    ),
    "recall": Recall(
        average=True, is_multilabel=True, output_transform=binary_transform
    ),
}

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

log_interval = 100


@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(
        f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(trainloader)
    metrics = train_evaluator.state.metrics
    print(
        f"""
        Training Results - Epoch[{trainer.state.epoch}]
        \tAvg loss:  {metrics['loss']:.3f}"""
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(validloader)
    metrics = val_evaluator.state.metrics
    print(
        f"""Validation Results - Epoch[{trainer.state.epoch}]
        \tAvg loss: {metrics['loss']:.2f}
        \tPrecision: {metrics['precision']:.3f}
        \tRecall :   {metrics['recall']:.3f}"""
    )


def score_function(engine):
    prec = engine.state.metrics["precision"]
    reca = engine.state.metrics["recall"]
    return 2 / ((1 / prec) + (1 / reca))


model_checkpoint = ModelCheckpoint(
    f"/data/boo/SOURCE_ATTRIBUTION/checkpoints/microresnet_{args.phoneme}",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="f1",
    global_step_transform=global_step_from_engine(trainer),
    require_empty=False,
)

val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

trainer.run(trainloader, max_epochs=args.epochs)

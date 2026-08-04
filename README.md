# EA-Net

Test-time adaptation for EEG encoders under realistic deployment shift.

## Why

EEG decoders degrade when they leave the lab that trained them — different
amplifier, different cap, different subject, different day. Test-time
adaptation (TTA) is the standard answer to that problem in vision and speech,
but recent reviews of EEG foundation models note that its application to EEG
[remains underexplored](https://arxiv.org/abs/2507.11783). This repository is a
controlled harness for measuring whether it actually helps here.

## What it does

Trains a compact EEG encoder on a source split, corrupts a held-out split to
emulate deployment conditions, then streams that split through a TTA method and
scores it.

**Shifts** (`eanet shifts`) — additive white and pink noise, per-channel gain
mismatch, dead electrodes, consumer-headset montage reduction, sub-1 Hz
baseline drift, residual mains interference. Each takes a severity from 1 to 5.

**Methods** — `source` (no adaptation), `norm` (BatchNorm statistic
recalibration), `tent` (entropy minimisation on BN affine parameters), `sar`
(reliable-sample filtering plus sharpness-aware minimisation).

**Backbones** — EEGNet v4 and Shallow ConvNet.

## Install

```bash
pip install -e .
```

Requires Python >=3.10 and PyTorch >=2.2. CPU is fine at this scale.

## Use

```bash
# List available corruptions
eanet shifts

# Single condition
eanet eval --set shift.kind=channel_dropout --set shift.severity=4 \
           --set tta.method=tent --out runs/tent_dropout.json

# Grid over shift x severity x method, training the source model once
eanet sweep --shifts none noise channel_dropout montage \
            --methods source norm tent sar --severities 2 4 \
            --out runs/sweep.json
```

Any config field is overridable with `--set key.sub=value`; defaults live in
[configs/default.yaml](configs/default.yaml).

## Data

`data.source=physionet` pulls the PhysioNet EEG Motor Movement/Imagery dataset
through MNE — 109 subjects, 64 channels, no registration required. Runs 4/8/12
are the left-vs-right fist imagery runs. Subjects are the shift axis: train on
one group, test on another.

`data.source=synthetic` generates separable band-power data for smoke tests.
It is deliberately easy and saturates near 100% accuracy — use it to check the
plumbing, never to compare methods.

## Reading the results

Report `balanced_accuracy` and `collapse_ratio` next to `accuracy`, always.
Entropy minimisation can collapse the model onto a single class and still post
a respectable accuracy on an imbalanced test set; `collapse_ratio` near 1.0
means the accuracy beside it is meaningless.

Adaptation is **online and order-dependent** by default — the model keeps
adapting across the test stream, so the batch order seed is part of the
experiment. Set `tta.episodic=true` to reset before each batch and measure
single-batch adaptation in isolation.

## Design notes

The models are BatchNorm-heavy on purpose: `norm`, `tent` and `sar` all adapt
BN statistics and affine parameters, so a backbone without BN gives them
nothing to work with.

During adaptation the model runs in **eval** mode, not train mode. The
reference TENT implementation uses `train()`, but only to force BN onto batch
statistics — its ResNets have no dropout. Clearing `running_mean`/`running_var`
achieves the same thing in either mode, and eval keeps dropout off, which
matters because EEGNet is dropout-heavy and active dropout would randomise both
the adaptation gradient and the reported logits.

## Tests

```bash
pytest
```

## Status

Harness is complete and tested against synthetic data. Real-data results on
PhysioNet are not yet in.

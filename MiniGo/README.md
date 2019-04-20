# MiniGo

This is an implementation of [MiniGo] using [Swift for TensorFlow].
The implementation is adapted from the
[MLPerf reference model](https://github.com/mlperf/training/tree/master/reinforcement).

> Note: Due to temporary limitations, we only support inference (self-play)
> for now. We hope to soon lift these restrictions.

In order to run this code, first download a pre-trained checkpoint from the
MiniGo project. Then, compile the code and run it locally. Parameters like
"board size" and "simulations per move" are defined as constants at the top of
[`Sources/MiniGo/main.swift`](https://github.com/tensorflow/swift-models/blob/stable/MiniGo/Sources/MiniGo/main.swift)
and can be modified.

## Getting started

### Install Swift for TensorFlow

Building MiniGo requires a Swift for TensorFlow toolchain.
To get a toolchain, you can:

1. [Download a pre-built package](https://github.com/tensorflow/swift/blob/master/Installation.md).
2. [Compile a toolchain from source](https://github.com/apple/swift/tree/tensorflow#building-swift-for-tensorflow).

### Restore a MiniGo checkpoint

Currently, inference uses a pre-trained checkpoint. To download it, please
install [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install) and run
the following in the parent directory:

```sh
cd swift-models
mkdir -p MiniGoCheckpoint
gsutil cp 'gs://minigo-pub/v15-19x19/models/000939-heron.*' MiniGoCheckpoint/
```

### Usage

```sh
# Run inference (self-plays).
cd swift-models
swift run -Xlinker -ltensorflow -c release MiniGo
```

[Swift for TensorFlow]: https://www.tensorflow.org/swift
[MiniGo]: https://github.com/mlperf/training/blob/master/reinforcement/tensorflow/minigo

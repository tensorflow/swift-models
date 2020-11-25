# MiniGo

This is an implementation of [MiniGo] using [Swift for TensorFlow].
The implementation is adapted from the
[MLPerf reference model](https://github.com/mlperf/training/tree/master/reinforcement).

> Note: Due to temporary limitations, we only support inference (self-play)
> for now. We hope to soon lift these restrictions.

Parameters like "board size" and "simulations per move" are defined as constants at the top of
[`Sources/MiniGo/main.swift`](https://github.com/tensorflow/swift-models/blob/stable/MiniGo/Sources/MiniGo/main.swift)
and can be modified.

## Getting started

### Install Swift for TensorFlow

Building MiniGo requires a Swift for TensorFlow toolchain.
To get a toolchain, you can:

1. [Download a pre-built package](https://github.com/tensorflow/swift/blob/master/Installation.md).
2. [Compile a toolchain from source](https://github.com/apple/swift/tree/tensorflow#building-swift-for-tensorflow).

### Usage

```sh
# Run inference (self-plays).
cd swift-models
swift run -c release MiniGoDemo
```

[Swift for TensorFlow]: https://www.tensorflow.org/swift
[MiniGo]: https://github.com/mlperf/training/blob/master/reinforcement/tensorflow/minigo

# MiniGo

This folder is the Swift ([S4TF]) verison of the [MiniGo] game. It has been
reimplemented from the MLPerf reference model.

> Note: Due to temporary limitations, we only support inference (self-play)
> for now. We hope to soon lift these restrictions.

In order to use this code, first download a pre-trained set of weights from
the MiniGo project. Then compile the code and run it locally! You can change
around parameters like the board size, and the number of simulations per
move by modifying constants at the top of `Sources/MiniGo/main.swift`.

## To Get a MiniGo Checkpoint

Currently the inference uses a pre-trained checkpoint. To get it, run

    mkdir -p MiniGo
    gsutil cp gs://minigo-pub/v15-19x19/models/000939-heron.* MiniGo/

## To Compile and Run

To compile and run the code, make sure you have the latest S4TF toolchain (see
next section).

Then, follow the following steps:

    swift test
    swift run

## Swift for TensorFlow Toolchain

In order to use [Swift Package Manager][SwiftPM], a toolchain is required. You
can

1. download the pre-built package from [official
   site](https://github.com/tensorflow/swift/blob/master/Installation.md)
2. [compile the toolchain](https://github.com/apple/swift/tree/tensorflow#building-swift-for-tensorflow)
   on your own


[S4TF]: https://www.tensorflow.org/swift/
[SwiftPM]: https://swift.org/package-manager/
[MiniGo]: https://github.com/mlperf/training/blob/master/reinforcement/tensorflow/minigo


# MiniGo

This folder is the Swift ([S4TF]) verison of the [MiniGo] game.

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

## Profiling MiniGo

### Linux

You can profile the MiniGo binary on Linux using the `perf` tool. You will need
to first enable profiling as a user:

```
sudo bash -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
```

Compile the package in opt mode:

```
swift build -c release
```

Now you can run the binary under `perf` to collect results. You may need to run
from a temporary directory as SrcFS occasionaly struggles with writing perf log
files:

```
cd /tmp/
perf record -g /path/to/swift-models/MiniGo /path/to/swift-models/MiniGo/.build/x86_64-unknown-linux/release/MiniGo
# hit <ctrl-c> at any point to quit early
```

You can use `perf report` to inspect the results. For a graphical
representation, you can use gprof2dot:

```
pip install --user gprof2dot
perf script | swift-demangle | ~/.local/bin/gprof2dot -f perf | dot -Tsvg > ~/www/swift-minigo.svg

# Open a window
perf script | swift-demangle | ~/.local/bin/gprof2dot -f perf | dot -Tx11
```

Or create a flame graph from the perf output:

```
git clone https://github.com/brendangregg/FlameGraph
export PATH=$PATH:$HOME/pkg/FlameGraph/
perf script | ~/pkg/usr/bin/swift-demangle | stackcollapse-perf.pl | flamegraph.pl > ~/www/swift-flame.svg
```

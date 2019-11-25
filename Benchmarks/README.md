# Model benchmarks

Eventually, these will contain a series of benchmarks against a variety of models in the 
swift-models repository. The following benchmarks have been implemented:

- Training LeNet against the MNIST dataset
- Performing inference with LeNet using MNIST-sized random images

These benchmarks should provide a baseline to judge performance improvements and regressions in 
Swift for TensorFlow.

## Running benchmarks

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To run all benchmarks, type the following while in the swift-models directory:

```sh
swift run -c release Benchmarks measure-all
```

To run an an individual benchmark, use `measure`:

```sh
swift run -c release Benchmarks measure --benchmark <name> --training
swift run -c release Benchmarks measure --benchmark <name> --inference
```

To list all benchmarks and their default settings use `list-defaults`:

```sh
swift run -c release Benchmarks list-defaults
```

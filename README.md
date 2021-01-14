# Swift for TensorFlow Models

This repository contains many examples of how Swift for TensorFlow can be used to build machine
learning applications, as well as the models, datasets, and other components required to build them.
These examples are intended to demonstrate best practices for the use of 
[Swift for TensorFlow APIs](https://github.com/tensorflow/swift-apis) and act as end-to-end tests
to validate the function and performance of those APIs.

Active development occurs on the `main` branch, and that is kept current against the `main` branch
of [the Swift compiler](https://github.com/apple/swift) and the `main` branch of [the Swift for TensorFlow APIs](https://github.com/tensorflow/swift-apis).

For stable snapshots, use the ```tensorflow-xx``` branch that corresponds to the toolchain you are using from the [Swift for TensorFlow releases](https://github.com/tensorflow/swift/blob/master/Installation.md#releases).  For example, for the 0.12 release, use the ```tensorflow-0.12``` branch.

To learn more about Swift for TensorFlow development, please visit
[tensorflow/swift](https://github.com/tensorflow/swift).

## Examples

The examples within this repository are all designed to be run as standalone applications. The easiest way to do this is to use Swift Package Manager to build and run individual examples. This 
can be accomplished by changing to the root directory of the project and typing something like

```bash
swift run -c release [Example] [Options]
```

For Windows, an additional flag may be required:

```cmd
swift run -Xswiftc -use-ld=lld -c release [Example] [Options]
```

This will build and run a specific example in the release configuration. Due to significant
performance differences between debug and release builds in Swift, we highly recommend running the
examples from a release build. Some examples have additional command-line options, and those will
be described in the example's README.

The following is a catalog of the current examples, grouped by subject area, with links to their
location within the project. Each example should have documentation for what it is demonstrating
and how to use it.

### Image classification

- [A custom model training against CIFAR-10](Examples/Custom-CIFAR10)
- [LeNet-5 training against MNIST](Examples/LeNet-MNIST)
- [MobileNet V1 training against Imagenette](Examples/MobileNetV1-Imagenette)
- [MobileNet V2 training against Imagenette](Examples/MobileNetV2-Imagenette)
- [ResNet-56 training against CIFAR-10](Examples/ResNet-CIFAR10)
- [ResNet-50 training against ImageNet](Examples/ResNet50-ImageNet)
- [VGG-16 training against Imagewoof](Examples/VGG-Imagewoof)

### Text

- [BERT training against CoLA](Examples/BERT-CoLA)
- [Pretrained GPT-2 performing text generation](Examples/GPT2-Inference)
- [GPT-2 training against WikiText-2](Examples/GPT2-WikiText2)
- [WordSeg](Examples/WordSeg)

### Generative models

- [1-D Autoencoder](Autoencoder/Autoencoder1D)
- [2-D Autoencoder](Autoencoder/Autoencoder2D)
- [1-D Variational Autoencoder](Autoencoder/VAE1D)
- [CycleGAN](CycleGAN)
- [GAN](GAN)
- [DCGAN](DCGAN)
- [pix2pix](pix2pix)

### Reinforcement learning

- [Blackjack](Gym)
- [CartPole](Gym)
- [Catch](Catch)
- [FrozenLake](Gym)
- [MiniGo](MiniGo)

### Standalone

- [Differentiable Shallow Water PDE Solver](Examples/Shallow-Water-PDE)
- [Fast Style Transfer](FastStyleTransfer)
- [Fractals](Examples/Fractals)
- [Growing Neural Cellular Automata](Examples/GrowingNeuralCellularAutomata)
- [Neural Collaborative Filtering using MovieLens](Examples/NeuMF-MovieLens)
- [PersonLab Human Pose Estimator](PersonLab)
- [Regression using BostonHousing](Examples/Regression-BostonHousing)

## Components

Beyond examples that use Swift for TensorFlow, this repository also contains reusable components
for constructing machine learning applications. These components reside in modules that can be
imported into separate Swift projects and used by themselves.

These components provide standalone machine learning models, datasets, image loading and saving,
TensorBoard integration, and a training loop abstraction, among other capabilities.

The Swift for TensorFlow models repository has acted as a staging ground for experimental
capabilities, letting us evaluate new components and interfaces before elevating them into the core
Swift for TensorFlow APIs. As a result, the design and interfaces of these components may change
regularly.

### Models

Several modules are provided that contain reusable Swift models for image classification, text
processing, and more. These modules are used within the example applications to demonstrate the
capabilities of these models, but they can also be imported into many other projects.

#### Image classification

Many common image classification models are present within
[the ImageClassificationModels module](Models/ImageClassification). To use them within a Swift
project, add ImageClassificationModels as a dependency and import the module:

```swift
import ImageClassificationModels
```

- DenseNet121
- EfficientNet
- LeNet-5
- MobileNetV1
- MobileNetV2
- MobileNetV3
- ResNet
- ResNetV2
- ShuffleNetV2
- SqueezeNet
- VGG
- WideResNet
- Xception

#### Recommendation

Several recommendation models are present within
[the RecommendationModels module](Models/Recommendation). To use them within a Swift
project, add RecommendationModels as a dependency and import the module:

```swift
import RecommendationModels
```

- DLRM
- MLP
- NeuMF

#### Text

Several text models are present within
[the TextModels module](Models/Text). To use them within a Swift
project, add TextModels as a dependency and import the module:

```swift
import TextModels
```

- [BERT](Models/Text/BERT)
- [GPT-2](Models/Text/GPT2)
- [WordSeg](Models/Text/WordSeg)

### Datasets

In addition to the machine learning model itself, a dataset is usually required to train the model.
Swift wrappers have been built for many common datasets to ease their use within machine learning
applications. Most of these use the
[Epochs](https://github.com/tensorflow/swift-apis/tree/main/Sources/TensorFlow/Epochs) API that 
provides a generalized abstraction of common dataset operations.

The [Datasets](Datasets) module provides these wrappers. To use them within a Swift
project, add Datasets as a dependency and import the module:

```swift
import Datasets
```

These are the currently provided dataset wrappers:

- [BostonHousing](Datasets/BostonHousing)
- [CIFAR-10](Datasets/CIFAR10)
- [MS COCO](Datasets/COCO)
- [CoLA](Datasets/CoLA)
- [ImageNet](Datasets/Imagenette)
- [Imagenette](Datasets/Imagenette)
- [Imagewoof](Datasets/Imagenette)
- [FashionMNIST](Datasets/MNIST)
- [KuzushijiMNIST](Datasets/MNIST)
- [MNIST](Datasets/MNIST)
- [MovieLens](Datasets/MovieLens)
- [Oxford-IIIT Pet](Datasets/OxfordIIITPets)
- [WordSeg](Datasets/WordSeg)

### Model checkpoints

Model saving and loading is provided by the [Checkpoints](Checkpoints). To use the model
checkpointing functionality, add Checkpoints as a dependency and import the module:

```swift
import Checkpoints
```

### Image loading and saving

The [ModelSupport](Support) module contains many shared utilites that are needed within the Swift
machine learning examples. This includes 

Experimental support for libjpeg-turbo as an accelerated image loader [is present](ImageLoader), 
but has not yet been incorporated into the main image loading capabilities.

### Generalized training loop

A generalized training loop that can be customized via callbacks is provided within the 
[TrainingLoop](TrainingLoop) module. All of the image classification examples use this training
loop, with the exception of the Custom-CIFAR10 example that demonstrates how to define your own
training loop from scratch. Other examples are being gradually converted to use this training loop.

### TensorBoard integration

[TensorBoard](https://www.tensorflow.org/tensorboard) integration is provided in the
[TensorBoard](TensorBoard) module as a callback for the generalized training loop. TensorBoard 
lets you visualize the progress of your model as it trains by plotting model statistics as they
update, or to review the training process afterward.

The [GPT2-WikiText2](Examples/GPT2-WikiText2) example demonstrates how this can be used when
training your own models.

## Benchmarks and tests

A core goal of this repository is to validate the proper function of the Swift for TensorFlow APIs.
In addition to the models and end-to-end applications present within this project, a suite of
benchmarks and unit tests reside here.

The benchmarks are split into a core of functionality, the
[SwiftModelsBenchmarksCore](SwiftModelsBenchmarksCore) module, and a 
[Benchmarks](SwiftModelsBenchmarks) command-line application for running these benchmarks. Refer to
the [documentation](SwiftModelsBenchmarks) for how to run the benchmarks on your system.

The [unit tests](Tests) verify functionality within models, datasets and other components. To run
them using Swift Package Manager on macOS or Linux:

```bash
swift test
```

and to run them on Windows:

```cmd
swift test -Xswiftc -use-ld=lld -c debug
```

## Using CMake for Development

In addition to Swift Package Manager, CMake can be used to build and run Swift for TensorFlow
models.

### *Experimental* CMake Support

There is experimental support for building with CMake.  This can be used to cross-compile the models and the demo programs.

It is highly recommended that you use CMake 3.16 or newer to ensure that `-B`
and parallel builds function properly in the example commands below. To install
this version on Ubuntu, we recommend following the instructions at
[Kitware's apt repo](https://apt.kitware.com/).

**Prerequisite:** [Ninja build tool](https://ninja-build.org/). Find
installation commands for your favorite package manager
[here](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages).

macOS:

```
# Configure
cmake                                                              \
  -B /BinaryCache/tensorflow-swift-models                          \
  -D BUILD_TESTING=YES                                             \
  -D CMAKE_BUILD_TYPE=Release                                      \
  -D CMAKE_Swift_COMPILER=$(TOOLCHAINS=tensorflow xcrun -f swiftc) \
  -G Ninja                                                         \
  -S /SourceCache/tensorflow-swift-models
# Build
cmake --build /BinaryCache/tensorflow-swift-models
# Test
cmake --build /BinaryCache/tensorflow-swift-models --target test
```

Linux:

```
# Configure
cmake                                     \
  -B /BinaryCache/tensorflow-swift-models \
  -D BUILD_TESTING=NO                     \
  -D CMAKE_BUILD_TYPE=Release             \
  -D CMAKE_Swift_COMPILER=$(which swiftc) \
  -G Ninja                                \
  -S /SourceCache/tensorflow-swift-models
# Build
cmake --build /BinaryCache/tensorflow-swift-models
```

Windows:

```
set DEVELOPER_LIBRARY_DIR=%SystemDrive%/Library/Developer/Platforms/Windows.platform/Developer/Library
:: Configure
"%ProgramFiles%\CMake\bin\cmake.exe"                                                                                                                                                   ^
  -B %SystemDrive%/BinaryCache/tensorflow-swift-models                                                                                                                                 ^
  -D BUILD_SHARED_LIBS=YES                                                                                                                                                             ^
  -D BUILD_TESTING=YES                                                                                                                                                                 ^
  -D CMAKE_BUILD_TYPE=Release                                                                                                                                                          ^
  -D CMAKE_Swift_COMPILER=%SystemDrive%/Library/Developer/Toolchains/unknown-Asserts-development.xctoolchain/usr/bin/swiftc.exe                                                        ^
  -D CMAKE_Swift_FLAGS="-sdk %SDKROOT% -I %DEVELOPER_LIBRARY_DIR%/XCTest-development/usr/lib/swift/windows/x86_64 -L %DEVELOPER_LIBRARY_DIR%/XCTest-development/usr/lib/swift/windows" ^
  -G Ninja                                                                                                                                                                             ^
  -S %SystemDrive%/SourceCache/tensorflow-swift-models
:: Build
"%ProgramFiles%\CMake\bin\cmake.exe" --build %SystemDrive%/BinaryCache/tensorflow-swift-models
:: Test
"%ProgramFiles%\CMake\bin\cmake.exe" --build %SystemDrive%/BinaryCache/tensorflow-swift-models --target test
```

## Bugs

Please report model-related bugs and feature requests using GitHub issues in
this repository.

## Community

Discussion about Swift for TensorFlow happens on the
[swift@tensorflow.org](https://groups.google.com/a/tensorflow.org/d/forum/swift)
mailing list.

## Contributing

We welcome contributions: please read the [Contributor Guide](CONTRIBUTING.md)
to get started. It's always a good idea to discuss your plans on
[the mailing list](https://groups.google.com/a/tensorflow.org/d/forum/swift) before making
any major submissions.

We have labeled some issues as ["good first issue"](https://github.com/tensorflow/swift-models/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
or ["help wanted"](https://github.com/tensorflow/swift-models/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
to provide some suggestions for where new contributors might be able to start.

## Code of Conduct

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

The Swift for TensorFlow community is guided by our [Code of
Conduct](CODE_OF_CONDUCT.md), which we encourage everybody to read before
participating.

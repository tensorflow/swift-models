# Swift for TensorFlow Models

This repository contains TensorFlow models written in Swift.

Use the ```tensorflow-xx``` branch that corresponds to the release you are using from [Swift for TensorFlow releases](https://github.com/tensorflow/swift/blob/master/Installation.md#releases).  For example, for the 0.6 release, use the ```tensorflow-0.6``` branch.

Actual development occurs on the `master` branch.
As new packages are released, `master` is pushed to `stable`.

For general information about Swift for TensorFlow development, please visit
[tensorflow/swift](https://github.com/tensorflow/swift).

## Development

### macOS and Linux

Use Swift Package Manager to develop Swift for TensorFlow models.

#### Build

```bash
swift build
```

#### Test

```bash
swift test
```

### Windows

Use CMake to develop Swift for TensorFlow models.

### *Experimental* CMake Support

There is experimental support for building with CMake.  This is required to build the models on Windows, and can also be used to cross-compile the models and the demo programs.

**NOTE**: tests are currently not supported with the CMake based build.

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
set SDKROOT=%SystemDrive%/Library/Developer/Platforms/Windows.platform/Developer/SDKs/Windows.sdk
set DEVELOPER_LIBRARY_DIR=%SystemDrive%/Library/Developer/Platforms/Windows.platform/Developer/Library
: Configure
"%ProgramFiles%\CMake\bin\cmake.exe"                                                                                                                                                                                                                  ^
  -B %SystemDrive%/BinaryCache/tensorflow-swift-models                                                                                                                                                                                                ^
  -D BUILD_SHARED_LIBS=YES                                                                                                                                                                                                                            ^
  -D BUILD_TESTING=YES                                                                                                                                                                                                                                ^
  -D CMAKE_BUILD_TYPE=Release                                                                                                                                                                                                                         ^
  -D CMAKE_Swift_COMPILER=%SystemDrive%/Library/Developer/Toolchains/unknown-Asserts-development.xctoolchain/usr/bin/swiftc.exe                                                                                                                       ^
  -D CMAKE_Swift_FLAGS="-sdk %SDKROOT% -I %SDKROOT%/usr/lib/swift -L %SDKROOT%/usr/lib/swift/windows -I %DEVELOPER_LIBRARY_DIR%/XCTest-development/usr/lib/swift/windows/x86_64 -L %DEVELOPER_LIBRARY_DIR%/XCTest-development/usr/lib/swift/windows " ^
  -G Ninja                                                                                                                                                                                                                                            ^
  -S %SystemDrive%/SourceCache/tensorflow-swift-models
: Build
"%ProgramFiles%\CMake\bin\cmake.exe" --build %SystemDrive%/BinaryCache/tensorflow-swift-models
: Test
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
to get started. It's always a good idea to discuss your plans on the mailing
list before making any major submissions.

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

# Transformer

This is an implementation of [OpenAI's GPT-2 Transformer language model](https://github.com/openai/gpt-2) using [Swift for TensorFlow](https://github.com/tensorflow/swift).

Currently, the model must be run from the root of the swift-models project directory. You can run 
the model by sampling either unconditionally:

## Building and Running

NOTE the first run will take a little while as it will download the GPT-2 data model from the internet.

A "temperature" of 0 means "always output the same text, but it'll be fairly boring,"
a temperature of 1 means sampling exactly according to the model probabilities, and a temperature
higher than 1 means sampling more randomly than the model probabilities indicate. Values of 0.5-0.8 tend
to be best.

### macOS and Linux

```sh
swift run -c release TransformerDemo [temperature]
```

or conditionally:

```sh
swift run -c release TransformerDemo [temperature] "conditioning text"
```

### Windows

Use CMake to develop Swift for TensorFlow models on Windows.

Although all the models build and run, not all of them have been tested.  Particularly, the automated download and extraction may not fully function on all environments.  The transformer model has been tested and is known to fully work on Windows.  Note that these operations must be performed from the `x64 Native Tools Command Prompt for VS2019` (it does not need to be run as Administrator).  This is **not** the same as `Command Prompt`, and is only available after Visual Studio has been installed.

#### Configure

Ensure that your
[installation](https://github.com/tensorflow/swift/blob/master/Installation.md#installation-2)
is up-to-date. In particular, ensure that you have deployed Windows SDK
modulemaps since your last Visual Studio update.

```cmd
git clone git://github.com/tensorflow/swift-models swift-models
set SDKROOT=%SystemDrive%/Library/Developer/Platforms/Windows.platform/Developer/SDKs/Windows.sdk
set SWIFTFLAGS=-sdk %SDKROOT% -I %SDKROOT%/usr/lib/swift -L %SDKROOT%/usr/lib/swift/windows -Xlinker -ignore:4217 -Xlinker -ignore:4286
"%ProgramFiles%/CMake/bin/cmake.exe"    ^
  -B build/swift-models                 ^
  -D BUILD_SHARED_LIBS=YES              ^
  -D CMAKE_BUILD_TYPE=Release           ^
  -D CMAKE_Swift_FLAGS="%SWIFTFLAGS%"   ^
  -G Ninja                              ^
  -S swift-models
```

#### Build

```cmd
cmake --build build/swift-models --target TransformerUI
```

#### Run

```cmd
md build\TransformerUI
copy build\swift-models\swift-protobuf-prefix\src\swift-protobuf-build\Sources\SwiftProtobuf\SwiftProtobuf.dll build\TransformerUI\
copy build\swift-models\swift-win32-prefix\src\swift-win32-build\SwiftWin32.dll build\TransformerUI\
copy build\swift-models\Batcher\Batcher.dll build\TransformerUI\
copy build\swift-models\Datasets\Datasets.dll build\TransformerUI\
copy build\swift-models\Models\Text\TextModels.dll build\TransformerUI\
copy build\swift-models\Support\ModelSupport.dll build\TransformerUI\
copy build\swift-models\Transformer\Transformer.dll build\TransformerUI\
copy build\swift-models\Transformer\TransformerUI.exe build\TransformerUI\
copy build\swift-models\Transformer\TransformerUI.exe.manifest build\TransformerUI\
```

Once all the files have been copied, you should be able to run the demo
application by either double clicking the `TransformerUI` executable from
Windows Explorer or running `build\TransformerUI\TransformerUI` from the
command line.

### Sample Run

Here's one output we got:

```console
$ swift run -c release TransformerDemo 0.5 "Introducing Swift for TensorFlow"
Introducing Swift for TensorFlow

Swift has been around since the beginning. It was created by the Swift team to enable developers to write Swift code. It is a powerful language for developing many different types of data structures.

In this tutorial, we will show you how to use Swift to write a simple, simple TensorFlow program.
```

### Dependencies

This code requires a Swift for TensorFlow toolchain.
To get a toolchain, you can:

1. [Download a pre-built package](https://github.com/tensorflow/swift/blob/master/Installation.md).
2. [Compile a toolchain from source](https://github.com/apple/swift/tree/tensorflow#building-swift-for-tensorflow).

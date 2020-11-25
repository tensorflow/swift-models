# GPT-2 Inference

This example demonstrates how to generate sequences of text using a
pre-trained
[OpenAI GPT-2 Transformer language model](https://github.com/openai/gpt-2).

A pre-trained GPT-2 network is instantiated from the library of standard models
and used to generate a sequence of text.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

### macOS and Linux

To generate text by sampling unconditionally, run:

```console
cd swift-models
swift run -c release GPT2-Inference
```

To generate text based on a provided seed, run:

```console
cd swift-models
swift run -c release GPT2-Inference [temperature] ["conditioning text"]
```

A "temperature" of 0 means "always output the same text, but it'll be fairly
boring," a temperature of 1 means sampling exactly according to the model
probabilities, and a temperature higher than 1 means sampling more randomly
than the model probabilities indicate. Values of 0.5-0.8 tend to be best.

Here's an example of expected output:

```console
$ swift run -c release GPT2-Inference 0.5 "Introducing Swift for TensorFlow"
Introducing Swift for TensorFlow

Swift has been around since the beginning. It was created by the Swift team to enable developers to write Swift code. It is a powerful language for developing many different types of data structures.

In this tutorial, we will show you how to use Swift to write a simple, simple TensorFlow program.
```

### Windows

Use CMake to develop Swift for TensorFlow models on Windows.

Although all the models build and run, not all of them have been tested.  Particularly, the automated download and extraction may not fully function on all environments.  The transformer model has been tested and is known to fully work on Windows.  Note that these operations must be performed from the `x64 Native Tools Command Prompt for VS2019` (it does not need to be run as Administrator).  This is **not** the same as `Command Prompt`, and is only available after Visual Studio has been installed.

#### Configure

Ensure that your
[installation](https://github.com/tensorflow/swift/blob/master/Installation.md#installation-2)
is up-to-date. In particular, ensure that you have deployed Windows SDK
modulemaps since your last Visual Studio update.

```console
git clone git://github.com/tensorflow/swift-models %SystemDrive%/SourceCache/swift-models
set SDKROOT=%SystemDrive%/Library/Developer/Platforms/Windows.platform/Developer/SDKs/Windows.sdk
set SWIFTFLAGS=-sdk %SDKROOT% -I %SDKROOT%/usr/lib/swift -L %SDKROOT%/usr/lib/swift/windows -Xlinker -ignore:4217 -Xlinker -ignore:4286
"%ProgramFiles%/CMake/bin/cmake.exe"        ^
  -B %SystemDrive%/BinaryCache/swift-models ^
  -D BUILD_SHARED_LIBS=YES                  ^
  -D CMAKE_BUILD_TYPE=Release               ^
  -D CMAKE_Swift_FLAGS="%SWIFTFLAGS%"       ^
  -G Ninja                                  ^
  -S %SystemDrive%/SourceCache/swift-models
```

#### Build

```console
cmake --build %SystemDrive%/BinaryCache/swift-models --target GPT2InferenceUI
```

#### Run

```console
md %SystemDrive%\BinaryCache\GPT2InferenceUI
copy %SystemDrive%\BinaryCache\swift-models\swift-protobuf-prefix\src\swift-protobuf-build\Sources\SwiftProtobuf\SwiftProtobuf.dll %SystemDrive%\BinaryCache\GPT2InferenceUI\
copy %SystemDrive%\BinaryCache\swift-models\swift-win32-prefix\src\swift-win32-build\SwiftWin32.dll %SystemDrive%\BinaryCache\GPT2InferenceUI\
copy %SystemDrive%\BinaryCache\swift-models\Batcher\Batcher.dll %SystemDrive%\BinaryCache\GPT2InferenceUI\
copy %SystemDrive%\BinaryCache\swift-models\Datasets\Datasets.dll %SystemDrive%\BinaryCache\GPT2InferenceUI\
copy %SystemDrive%\BinaryCache\swift-models\Models\Text\TextModels.dll %SystemDrive%\BinaryCache\GPT2InferenceUI\
copy %SystemDrive%\BinaryCache\swift-models\Support\ModelSupport.dll %SystemDrive%\BinaryCache\GPT2InferenceUI\
copy %SystemDrive%\BinaryCache\swift-models\Examples\GPT2-Inference\GPT2InferenceUI.exe %SystemDrive%\BinaryCache\GPT2InferenceUI\
copy %SystemDrive%\BinaryCache\swift-models\Examples\GPT2-Inference\GPT2InferenceUI.exe.manifest %SystemDrive%\BinaryCache\GPT2InferenceUI\
```

Once all the files have been copied, you should be able to run the demo
application by either double clicking the `GPT2InferenceUI` executable from
Windows Explorer or running `build\GPT2InferenceUI\GPT2InferenceUI` from the
command line.


# MiniGo

This is an implementation of [OpenAI's GPT-2 Transformer language model](github.com/openai/gpt-2) using [Swift for TensorFlow](github.com/tensorflow/swift).

In order to run this code, first download a pre-trained checkpoint from OpenAI
using the included `download_model.sh` script. Then, compile and run:

```sh
bash download_model.sh
swift -O Operators.swift Model.swift PythonCheckpointReader.swift main.swift
```

This code requires a Swift for TensorFlow toolchain.
To get a toolchain, you can:

1. [Download a pre-built package](https://github.com/tensorflow/swift/blob/master/Installation.md).
2. [Compile a toolchain from source](https://github.com/apple/swift/tree/tensorflow#building-swift-for-tensorflow).

It also currently requires Python 3.x.

Both the tokenizer (`encoder.py`) and the model download script (`download_model.sh`) are
from the OpenAI implementation and are licensed under MIT.

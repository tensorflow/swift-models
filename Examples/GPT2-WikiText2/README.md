# GPT-2 with WikiText2

This example demonstrates how to fine-tune the [GPT-2](https://github.com/openai/gpt-2) network on the [WikiText2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

A pre-trained GPT-2 network is instantiated from the library of standard models, and applied to an instance of the WikiText dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release GPT2-WikiText2
```

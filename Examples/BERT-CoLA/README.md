# BERT with CoLA

This example demonstrates how to retrain a [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) model on the [CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability) dataset.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release BERT-CoLA
```

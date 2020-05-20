# Word Segmentation

This example demonstrates how to train the [word segmentation model][model]
against the dataset provided in the paper
["Learning to Discover, Ground, and Use Words with Segmental Neural Language
Models"][paper]
by Kazuya Kawakami, Chris Dyer, and Phil Blunsom.

A segmental neural language model (SNLM) is instantiated from the library of
standard models. A custom training loop is defined and the training
losses for each epoch are shown.

This implementation is not affiliated with DeepMind and has not been verified by
the authors.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow][s4tf] installed. Make sure you've added the correct version of
`swift` to your path.

To train the model using the full datasets published in the paper, run:

```sh
cd swift-models
swift run -c release WordSeg
```

To train the model using a smaller, unrealistic sample dataset, run:

```sh
cd swift-models
swift run -c release WordSeg Examples/WordSeg/smalldata.txt
```

To run the model with your own dataset, run:

```sh
cd swift-models
swift run -c release WordSeg path/to/training_data.txt [path/to/validation_data.txt [path/to/test_data.txt]]
```

[model]: https://github.com/tensorflow/swift-models/tree/master/Models/Text/WordSeg
[paper]: https://www.aclweb.org/anthology/P19-1645.pdf
[s4tf]: https://github.com/tensorflow/swift/blob/master/Installation.md

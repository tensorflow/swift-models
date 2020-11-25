# Word Segmentation

This example demonstrates how to train the [word segmentation model][model]
against the dataset provided in the paper
["Learning to Discover, Ground, and Use Words with Segmental Neural Language
Models"][paper]
by Kazuya Kawakami, Chris Dyer, and Phil Blunsom.

A segmental neural language model (SNLM) is instantiated from the library of
standard models. A custom training loop is defined and the training
losses for each epoch are shown. For an explanation of this approach, see
the [Understanding WordSeg doc][understanding].

This implementation is not affiliated with DeepMind and has not been verified by
the authors.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow][s4tf] installed. Make sure you've added the correct version of
`swift` to your path.

## Execution

To train the model to accuracy using the full datasets published in the paper, run:

```sh
cd swift-models
swift run -c release WordSeg
```

To train the model using a smaller, unrealistic sample dataset, run:

```sh
swift run -c release WordSeg \
  --training-path Examples/WordSeg/smalldata.txt \
  --validation-path Examples/WordSeg/smalldata.txt
```

To run the model with your own dataset, run:

```sh
swift run -c release WordSeg \
  --training-path path/to/training_data.txt \
  [ --validation-path path/to/validation_data.txt \
  [ --test-path path/to/test_data.txt ]]
```

To view a list of all configurable parameters and their defaults, run:

```sh
swift run -c release WordSeg --help
```

[model]: https://github.com/tensorflow/swift-models/tree/master/Models/Text/WordSeg
[paper]: https://www.aclweb.org/anthology/P19-1645.pdf
[s4tf]: https://github.com/tensorflow/swift/blob/master/Installation.md
[understanding]: https://docs.google.com/document/d/1NlFH0_89gB_qggtgzJIKYHL2xPI3IQjWjv18pnT1M0E

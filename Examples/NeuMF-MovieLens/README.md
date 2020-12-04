# TopK Recommendation System using Neural Collaborative Filtering

This example demonstrates how to train a recommendation system with implicit feedback on the
MovieLens 100K (ml-100k) dataset using a [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
model. This model trains on binary information about whether or not a user interacted with a specific item.
To target the models for an implicit feedback and ranking task, we optimize them using sigmoid cross entropy
loss with negative sampling.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/main/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.
To train the model, run:

```sh
cd swift-models
swift run NeuMF-MovieLens
```

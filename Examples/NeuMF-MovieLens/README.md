# TopK Recommendation System using Neural Collaborative Filtering

This example demonstrates how to train recommendation system with implicit feedback on the
MovieLens 100K (ml-100k) dataset with a Neural Collaborative Filtering model. This model
trains on a binary information about whether or not user interacted with specific item.
To target the models for implicit feedback and ranking task, we optimize them
using sigmoid cross entropy loss with negative sampling.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.
To train the model, run:

```sh
cd swift-models
swift run NeuMF-MovieLens
```

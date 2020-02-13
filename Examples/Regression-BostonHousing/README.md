# Regression with tabular Boston housing price dataset

This example demonstrates how to train a regression model against the [Boston 
housing price dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/).

There are 506 data points. Samples contain 13 attributes of houses at different 
locations around the Boston suburbs in the late 1970s. Targets are the median 
values of the houses at a location (in k$).

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed and added to your path. Additionally, the data loader requires Python
3.x, `wget`, and `numpy`.

> Note: For macOS, you need to set up the `PYTHON_LIBRARY` to help the Swift for
> TensorFlow find the `libpython3.<minor-version>.dylib` file, e.g., in
> `homebrew`.

To train the model, run:

```sh
cd swift-models
swift run Regression-BostonHousing
```

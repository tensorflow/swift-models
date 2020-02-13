# Tabular

This directory contains a regression model trains a Boston house prices dataset.

This is a very small dataset with only 506 data points and 13 numeric/categorical features.  Median Value (last column) is usually the target.

:Creator: Harrison, D. and Rubinfeld, D.L.
This is a copy of UCI ML housing dataset. https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

## Setup

You'll need [the latest version][INSTALL] of Swift for TensorFlow
installed and added to your path. Additionally, the data loader requires Python
3.x (rather than Python 2.7), `wget`, and `numpy`.

> Note: For macOS, you need to set up the `PYTHON_LIBRARY` to help the Swift for
> TensorFlow find the `libpython3.<minor-version>.dylib` file, e.g., in
> `homebrew`.

To train the model on the Boston house prices dataset, run:

```
cd swift-models
swift run Tabular
```

[INSTALL]: (https://github.com/tensorflow/swift/blob/master/Installation.md)

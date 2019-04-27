# Swift for TensorFlow Models

This repository contains TensorFlow models written in Swift.

The `stable` branch works with the latest [Swift for TensorFlow releases](https://github.com/tensorflow/swift/blob/master/Installation.md#releases).

Actual development occurs on the `master` branch.
As new packages are released, `master` is pushed to `stable`.

For general information about Swift for TensorFlow development, please visit
[tensorflow/swift](https://github.com/tensorflow/swift).

## Development

Use Swift Package Manager to develop Swift for TensorFlow models.

### Build

```bash
swift build -Xlinker -ltensorflow
```

### Test

```bash
swift test -Xlinker -ltensorflow
```

## Bugs

Please report model-related bugs and feature requests using GitHub issues in
this repository.

## Community

Discussion about Swift for TensorFlow happens on the
[swift@tensorflow.org](https://groups.google.com/a/tensorflow.org/d/forum/swift)
mailing list.

## Contributing

We welcome contributions: please read the [Contributor Guide](CONTRIBUTING.md)
to get started. It's always a good idea to discuss your plans on the mailing
list before making any major submissions.

## Code of Conduct

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

The Swift for TensorFlow community is guided by our [Code of
Conduct](CODE_OF_CONDUCT.md), which we encourage everybody to read before
participating.

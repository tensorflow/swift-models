## Imagenette

[Imagenette](https://github.com/fastai/imagenette) is a subset of the Imagenet dataset, designed to be a testbed for building real-world image classifiers.  This code demonstrates a basic loop of building an input/validation dataset from a set of directories, then builds a simple CNN model and trains/validates it using said input.

To get the data:

    wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz && tar -xvf imagenette-160.tgz



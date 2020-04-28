# LeNet-5 with MNIST

This example demonstrates how to train the [LeNet-5 network]( http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) against the [MNIST digit classification dataset](http://yann.lecun.com/exdb/mnist/) using the [XLA](https://www.tensorflow.org/xla) backend.

The LeNet network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the MNIST dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.


## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

You'll need to explictly set the XLA device or CPU fallback mode will be used.

For a local GPU, set the following parameters:

    export XRT_WORKERS='localservice:0;grpc://localhost:40934'
    export XRT_DEVICE_MAP="GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"

For a TPU, set the following parameters (your calling code will need to be on a local network):

    export XLA_USE_XRT=1
    export XRT_TPU_CONFIG="tpu_worker;0;<TPU_DEVICE_IP>:8470"
    export XRT_WORKERS='localservice:0;grpc://localhost:40934'
    export XRT_DEVICE_MAP="TPU:0;/job:localservice/replica:0/task:0/device:TPU:0"

Then, to train the model, run:

```sh
cd swift-models
swift run -c release LeNet-MNIST-XLA
```

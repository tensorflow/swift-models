## ImageNet + ResNet 50

This code demonstrates training a [ResNet50](https://arxiv.org/abs/1512.03385) model (specifically, the 1.5 variant) from scratch on the [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/index#cite) dataset.

### Data

ImageNet is a large dataset (you will need >300GB of free space to process it locally) and so you will currently have to download it yourself.  You can do so using the raw data files and then apply the post-procesing steps listed in ImageNet.swift yourself, or you can download a pre-processed archive directly like so:

    wget -O /tmp/imagenet.tgz https://REMOTE-SERVER/imagenet/imagenet.tgz

### Demo

After doing the above, you will be able to run this demo.  Here we implement a standard SGD training loop with decreasing learning rates + the standard imagenet data augmentation process (eg random crops + flips applied to our training data set).

# pix2pix

This example tries to closely resemble [the original pix2pix paper](https://arxiv.org/abs/1611.07004) with the same architecture and hyper-parameters, but using Swift for TensorFlow. Results seem to be close to what the original authors claimed to achieve, this is what a generator outputs after 7 hours of training using single GTX 1080 card:

![](https://i.imgur.com/Od9dfe8.jpg)
![](https://i.imgur.com/zQxPCAd.jpg)

In order to run the project you have to just call 

```bash
swift run -c release pix2pix
```

You will need a tensorboard installation in order to run the project. 


## Notes

- Trainer will emit a lot of intermediate results in a calling directory, you can easily comment out that part. It is a workaround because for some reason tensorboard crashes when provided with result images, I'm investigating on the problem.
- You need to specify a different URL for `dataset` or this will download the facades dataset for you.
- This implementation only provides a data loader for the facades dataset, but the model is capable of learning on any pix2pix compatible dataset.

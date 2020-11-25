# PersonLab

[PersonLab](https://arxiv.org/abs/1803.08225) human pose estimator, inference only version.

CLI demo with option to run benchmarks on a local image file. Very fast at over 200 fps on a GeForce GTX 1080 Ti, which it vastly underutilizes.

## Checkpoints
Download the [checkpoint](https://github.com/joaqo/swift-models/releases/download/PersonlabDemo/personlabCheckpoint.tar.gz) from the releases page in this repo, whose path you'll have to provide to the CLI demo.

## Running
```bash
swift run PersonLab --help
```

## Notes
- Compiling for release (`swift run -c release PersonLab`) makes the decoder run about 10 times faster.
- Had to build slightly custom mobilenet backbone as the checkpoint I used does not fit into the mobilenet versions available in this repo.
- First part of this [video](https://www.youtube.com/watch?v=WxFPrypPBpU) is a short summary of how the model works and an explanation of some key parts of the codebase.

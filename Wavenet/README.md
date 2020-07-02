# Wavenet

## Setup
This example trains a [Wavenet](https://arxiv.org/abs/1611.07004) using Swift for TensorFlow. The open source Python Tensrflow version [here](https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py) was used as a reference implementation for this example.

In order to run the project you'll need to install a few Python deps that are used to load data

```
pip install -r requirements.txt
```

You'll also need to install `ffmpeg`:
On Ubuntu
```
apt-get install ffmpeg
```
On MacOSX, use your favorite package manager. For brew:
```
brew install ffmpeg
```

To kick off model training, run:

```bash
swift run Wavenet
```

## Blockers
At the moment only the training loop works but there are some limitations / missing features that
need to be addressed:
- [ ] Add PaddingFIFOQueue to correctly produce batches from data files
- [ ] Add the generation step to generate new audio samples
- [ ] Add preprocessing function to trim silent sections of audio
- [ ] Make model configurable with command line args


## Notes
- As of now, only the unconditioned Audio Generation task is supported. We will add more tasks in follow ups:
	- Audio generation with Global Conditioning
	- Text to speech
        - Discriminative audio tasks like speech recognition
- By default, this example will download the [VCTK dataset](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
- This implementation only provides a data loader for the VCTK dataset at the moment but should work with any other audio dataet
  with minor modifications


## Follow-up improvements
- Use`librosa` instead of the simplistic `pydub` library in Python for reading and processing audio. Initial attempts resulted in
the following error:
```
Assertion failed: (PassInf && "Expected all immutable passes to be initialized"), function addImmutablePass, file /Users/buildbot/miniconda3/conda-bld/llvmdev_1556270736866/work/lib/IR/LegacyPassManager.cpp, line 849.
Abort trap: 6
```

- Possibly use Swift-native audio libraries like AVFoundation? Initial attempts at this caused linker issues:
```
dyld: Symbol not found $<symbol-name>
expected in <path-to-swift-toolchain>/AVFoundation.swift
```
- Add XLA support
- Add support for more tasks and datasets
- Add scalar input mode
- Add L2 reg

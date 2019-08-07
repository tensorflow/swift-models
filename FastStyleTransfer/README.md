# Fast Style Transfer

Based on the [PyTorch implementation](https://github.com/pytorch/examples/tree/master/fast_neural_style).
The model should be trainable, but so far it's only tested for inference with pre-trained weights (included in `Demo/weights`). 

## Example
Run demo application to apply styles to jpeg images:
```
swift run FastStyleTranserDemo --weights=FastStyleTranser/Demo/weights/candy.npz --input=FastStyleTranser/Demo/examples/cat.jpg --output=candy_cat.jpg
swift run FastStyleTranserDemo --weights=FastStyleTranser/Demo/weights/mosaic.npz --input=FastStyleTranser/Demo/examples/cat.jpg --output=mosaic_cat.jpg
```

<img src="Demo/examples/cat.jpg" height="240" width="240" align="left">
<img src="Demo/examples/cat_candy.jpg" height="240" width="240" align="left">
<img src="Demo/examples/cat_mosaic.jpg" height="240" width="240">

## Requirements
Requires Python and NumPy to load weights.

## Jupyter Notebook
Run [demo notebook](Demo/ColabDemo.ipynb) in [Colab](https://colab.research.google.com/github/vvmnnnkv/swift-models/blob/fast-style/FastStyleTransfer/Demo/ColabDemo.ipynb)!

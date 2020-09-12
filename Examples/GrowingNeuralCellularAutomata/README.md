# Growing Neural Cellular Automata

This example replicates the paper ["Growing Neural Cellular Automata"](https://distill.pub/2020/growing-ca/)
by Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin. Currently, 
Experiment 1 ("Learning to grow"), Experiment 2 ("What persists, exists"), and Experiment 3 ("Learning to regenerate")
have been implemented.

In this example, cellular automata with continuous state values use an update rule dictated 
by a small neural network. The network in charge of the update rule is trained to cause the
cells to grow from a single cell into the shape and colors of the target image. The alpha
channel of the input image determines the shape of the image, and any cells with an alpha
less than 0.1 are considered "dead".

This is an example of results of Experiment 1, where the cells grow into the target image (a lizard emoji, in this case)
and then continue growing in an unbounded manner:

<img src="images/lizard-growth.gif" height="72" width="72" align="left">

In Experiment 2, the cells are trained to stabilize at the final image:

<img src="images/lizard-persistence.gif" height="72" width="72" align="left">

In Experiment 3, the cells are trained to regenerate portions of the image that have been damaged:

<img src="images/lizard-regrowth.gif" height="72" width="72" align="left">

During the growing phase of inference, a single cell at the center of the image is seeded with a
1.0 alpha channel, with all other values set to 0.0. Images are captured at multiple steps to
observe the evolution of the environment. During the regrowth phase, half of the image is cut away
and the steps are recorded as the cells attempt to regenerate the missing portion.

Representative images of the final state will be written into `output/`, with names
like `iteration[step].png`. Inference will write out one `step[number].png` frame into `output/`
for each time step, followed by `step_regen[number].png` for the regrowth phase.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the cell update rule per Experiment 1 ("Learning to grow"), use the following:

```sh
cd swift-models
swift run -c release GrowingNeuralCellularAutomata --image examples/GrowingNeuralCellularAutomata/images/lizard.png
```

For Experiment 2 ("What persists, exists"), a sample pool needs to be used:

```sh
swift run -c release GrowingNeuralCellularAutomata --use-sample-pool --image examples/GrowingNeuralCellularAutomata/images/lizard.png
```

For Experiment 3 ("Learning to regenerate"), both a sample pool and damage during training
need to be applied:

```sh
swift run -c release GrowingNeuralCellularAutomata --use-sample-pool --damaged-samples 3 --image examples/GrowingNeuralCellularAutomata/images/lizard.png
```

Parameters:

- `--image`: The path to the image that will be used as the desired target for the cellular automata.
- `--eager`, `--x10`: Whether to use the eager-mode or X10 backend (default: eager).
- `--image-size`: The height and width to use when resizing the input image (default: 40).
- `--padding`: The padding to add around the input image after resizing (default: 16).
- `--state-channels`: The number of state channels for each cell (default: 16).
- `--batch-size`: The batch size during training (default: 8).
- `--cell-fire-rate`: The fraction of cells to fire at each update (default: 0.5).
- `--use-sample-pool`: Whether to use a sample pool. 
- `--pool-size`: The pool size during training (default: 1024). 
- `--iterations`: The number of training iterations (default: 8000).
- `--damaged-samples`: The number of samples per batch to apply circular damage on (default: 0).

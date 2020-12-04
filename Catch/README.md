# Catch

This directory builds a reinforcement agent for the game Catch.

**Note:** This model is a work in progress and training doesn't quite work.
Specific areas for improvement are listed at the top of `catch.swift`.

Catch (adapted from [Mnih et al., 2014](https://arxiv.org/pdf/1406.6247.pdf))
is played on a 5x5 grid of binary pixels. There are only two objects: a pixel
representing a ball falling from the top of the grid and a pixel representing a
paddle at the bottom of the grid. At each time step, the agent can choose to
move the paddle left one pixel, right one pixel, or take no action.

When the falling ball reaches the bottom of the grid, the agent gets a reward
of 1 if the ball overlaps with the paddle, or -1 otherwise.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/main/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```
swift -O main.swift
```

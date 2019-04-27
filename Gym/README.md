# OpenAI Gym

This directory contains reinforcement learning algorithms in [OpenAI Gym](https://gym.openai.com) environments.

## [CartPole](https://gym.openai.com/envs/CartPole-v0)

> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

## [FrozenLake](https://gym.openai.com/envs/FrozenLake-v0)

> The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

Please install OpenAI Gym to run these models.
```bash
pip install gym
```

To build and run the models, run:

```bash
swift run CartPole
swift run FrozenLake
```

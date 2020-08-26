// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

/// The actor network that returns a probability for each action.
///
/// Actor-Critic methods has an actor network and a critic network. The actor network is the policy
/// of the agent: it is used to select actions.
struct ActorNetwork: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1, l2, l3: Dense<Float>

    init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
        l1 = Dense<Float>(
            inputSize: observationSize,
            outputSize: hiddenSize,
            activation: tanh,
            weightInitializer: heNormal()
        )
        l2 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: hiddenSize,
            activation: tanh,
            weightInitializer: heNormal()
        )
        l3 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: actionCount,
            activation: softmax,
            weightInitializer: heNormal()
        )
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2, l3)
    }
}

/// The critic network that returns the estimated value of each action, given a state.
///
/// Actor-Critic methods has an actor network and a critic network. The critic network is used to
/// estimate the value of the state-action pair. With these value functions, the critic can evaluate
/// the actions made by the actor.
struct CriticNetwork: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var l1, l2, l3: Dense<Float>

    init(observationSize: Int, hiddenSize: Int) {
        l1 = Dense<Float>(
            inputSize: observationSize,
            outputSize: hiddenSize,
            activation: relu,
            weightInitializer: heNormal()
        )
        l2 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: hiddenSize,
            activation: relu,
            weightInitializer: heNormal()
        )
        l3 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: 1,
            weightInitializer: heNormal()
        )
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2, l3)
    }
}

/// The actor-critic that contains actor and critic networks for action selection and evaluation.
///
/// Weight are often shared between the actor network and the critic network, but in this example,
/// they are separated networks.
struct ActorCritic: Layer {
    var actorNetwork: ActorNetwork
    var criticNetwork: CriticNetwork

    init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
        self.actorNetwork = ActorNetwork(
            observationSize: observationSize,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        self.criticNetwork = CriticNetwork(
            observationSize: observationSize,
            hiddenSize: hiddenSize
        )
    }

    @differentiable
    func callAsFunction(_ state: Tensor<Float>) -> Categorical<Int32> {
        precondition(state.rank == 2, "The input must be 2-D ([batch size, state size]).")
        let actionProbs = self.actorNetwork(state).flattened()
        let dist = Categorical<Int32>(probabilities: actionProbs)
        return dist
    }
}

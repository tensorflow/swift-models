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

struct ActorNet: Layer {
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

struct CriticNet: Layer {
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

class ActorCritic {
    var actorNet: ActorNet
    var criticNet: CriticNet

    init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
        self.actorNet = ActorNet(
            observationSize: observationSize,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        self.criticNet = CriticNet(
            observationSize: observationSize,
            hiddenSize: hiddenSize
        )
    }

    func act(state: Tensor<Float>, memory: PPOMemory) -> Int32 {
        // Input to the network needs to be 2D (BATCH_SIZE x STATE_SIZE)
        let state = Tensor<Float>([state])
        let actionProbs = self.actorNet(state).flattened()
        let dist = Categorical<Int32>(probabilities: actionProbs)
        let action = dist.sample().scalarized()
        let logProb = dist.logProbabilities.makeNumpyArray()[action]

        let convertedStates: [Float] = Array(numpy: state.makeNumpyArray().flatten())!
        memory.states.append(convertedStates)
        memory.actions.append(action)
        memory.logProbs.append(Float(logProb)!)

        return action
    }
}

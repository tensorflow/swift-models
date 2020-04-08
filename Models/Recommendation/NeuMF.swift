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
/// NeuMF is a recommendation model that combines matrix factorization and a multi-layer perceptron.
///
/// Original Paper:
/// "Neural Collaborative Filtering"
/// Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua
/// https://arxiv.org/pdf/1708.05031.pdf

public struct NeuMF: Module {

    public typealias Scalar = Float
    @noDerivative public let numUsers: Int
    @noDerivative public let numItems: Int
    @noDerivative public let mfDim: Int
    @noDerivative public let mfReg: Scalar
    @noDerivative public var mlpLayerSizes : [Int] = [64,32,16,8]
    @noDerivative public var mlpLayerRegs: [Scalar] = [0,0,0,0]

    public var mfUserEmbed: Embedding<Scalar>
    public var mfItemEmbed: Embedding<Scalar>
    public var mlpUserEmbed: Embedding<Scalar>
    public var mlpItemEmbed: Embedding<Scalar>
    public var dense1: Dense<Scalar>
    public var dense2: Dense<Scalar>
    public var dense3: Dense<Scalar>
    public var finalDense: Dense<Scalar>

    /// Initializes a NeuMF model as per the dataset from the given hyperparameters.
    ///
    /// -Parameters
    /// - numUsers: Total number of users in the dataset.
    /// - numItems: Total number of items in the dataset.
    /// - mfDim: Embedding size of the matrix factorization model.
    /// - mfReg: Regularization for the matrix factorization embeddings.
    /// - mlpLayerSizes: The sizes of the layers in the multi-layer perceptron model.
    /// - mlpLayerRegs: Regularization for each multi-layer perceptron layer.
    ///
    ///  Note: The first MLP layer is the concatenation of user and item embeddings, so mlpLayerSizes[0]/2 is the embedding size.

    public init(
        numUsers: Int,
        numItems: Int,
        mfDim: Int,
        mfReg: Float,
        mlpLayerSizes: [Int],
        mlpLayerRegs: [Float]
    ) {
        self.numUsers = numUsers
        self.numItems = numItems
        self.mfDim = mfDim
        self.mfReg = mfReg
        self.mlpLayerSizes = mlpLayerSizes
        self.mlpLayerRegs = mlpLayerRegs

        precondition(mlpLayerSizes[0]%2 == 0, "Input of first MLP layers must be multiple of 2")
        precondition(mlpLayerSizes.count == mlpLayerRegs.count, "Size of MLP layers and MLP reqularization must be equal")

        // TODO: regularization
        // Embedding Layer
        self.mfUserEmbed = Embedding<Scalar>(vocabularySize: self.numUsers, embeddingSize: self.mfDim)
        self.mfItemEmbed = Embedding<Scalar>(vocabularySize: self.numItems, embeddingSize: self.mfDim)
        self.mlpUserEmbed = Embedding<Scalar>(vocabularySize: self.numUsers, embeddingSize: self.mlpLayerSizes[0]/2)
        self.mlpItemEmbed = Embedding<Scalar>(vocabularySize: self.numItems, embeddingSize: self.mlpLayerSizes[0]/2)

        // TODO: Extend it for n layers by using for loop
        // Currently only for 4 layers
        dense1 = Dense(inputSize: self.mlpLayerSizes[0], outputSize: self.mlpLayerSizes[1], activation: relu)
        dense2 = Dense(inputSize: self.mlpLayerSizes[1], outputSize: self.mlpLayerSizes[2], activation: relu)
        dense3 = Dense(inputSize: self.mlpLayerSizes[2], outputSize: self.mlpLayerSizes[3], activation: relu)
        finalDense = Dense(inputSize: (self.mlpLayerSizes[3] + self.mfDim), outputSize: 1)
    }
        @differentiable
        public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar>{
            // Extracting user and item from dataset
            let userIndices = input.unstacked(alongAxis:1)[0]
            let itemIndices = input.unstacked(alongAxis:1)[1]

            // MLP part
            let userEmbedMlp = self.mlpUserEmbed(userIndices)
            let itemEmbedMlp = self.mlpItemEmbed(itemIndices)

            // MF part
            let userEmbedMf = self.mfUserEmbed(userIndices)
            let itemEmbedMf = self.mfItemEmbed(itemIndices)

            // Concatenate MF and MLP parts
            let mfVector = userEmbedMf*itemEmbedMf
            var mlpVector = userEmbedMlp.concatenated(with:itemEmbedMlp,alongAxis:-1)

            // Final prediction layer
            mlpVector = mlpVector.sequenced(through: dense1, dense2, dense3)
            let vector = mlpVector.concatenated(with:mfVector,alongAxis:-1)

            return finalDense(vector)
        }
}

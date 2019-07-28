// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TensorFlowModels",
    platforms: [
        .macOS(.v10_13)
    ],
    products: [
        .library(name: "ImageClassificationModels", targets: ["ImageClassificationModels"]),
        .library(name: "Datasets", targets: ["Datasets"]),
        .executable(name: "LeNet-MNIST", targets: ["LeNet-MNIST"]),
        .executable(name: "CIFAR", targets: ["CIFAR"]),
        .executable(name: "ResNet", targets: ["ResNet"]),
        .executable(name: "MiniGoDemo", targets: ["MiniGoDemo"]),
        .library(name: "MiniGo", targets: ["MiniGo"]),
        .executable(name: "GAN", targets: ["GAN"]),
    ],
    targets: [
        .target(name: "ImageClassificationModels", path: "Models/ImageClassification"),
        .target(name: "Datasets", path: "Datasets"),
        .target(name: "Autoencoder", dependencies: ["Datasets"], path: "Autoencoder"),
        .target(name: "CIFAR", path: "CIFAR"),
        .target(name: "Catch", path: "Catch"),
        .target(name: "Gym-FrozenLake", path: "Gym/FrozenLake"),
        .target(name: "Gym-CartPole", path: "Gym/CartPole"),
        .target(name: "Gym-Blackjack", path: "Gym/Blackjack"),
        .target(name: "LeNet-MNIST", dependencies: ["ImageClassificationModels", "Datasets"], 
				path: "Examples/LeNet-MNIST"),
        .target(name: "MiniGo", path: "MiniGo", exclude: ["main.swift"]),
        .target(name: "MiniGoDemo", dependencies: ["MiniGo"], path: "MiniGo",
                sources: ["main.swift"]),
        .testTarget(name: "MiniGoTests", dependencies: ["MiniGo"]),
        .target(name: "ResNet", path: "ResNet"),
        .target(name: "Transformer", path: "Transformer"),
        .target(name: "GAN", dependencies: ["Datasets"], path: "GAN"),
    ]
)

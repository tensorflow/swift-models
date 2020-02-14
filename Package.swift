// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TensorFlowModels",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library( name: "Batcher", targets: ["Batcher"]),
        .library(name: "Datasets", targets: ["Datasets"]),
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
        .library(name: "ImageClassificationModels", targets: ["ImageClassificationModels"]),
        .library(name: "TextModels", targets: ["TextModels"]),
        .executable(name: "VGG-Imagewoof", targets: ["VGG-Imagewoof"]),
        .executable(name: "Custom-CIFAR10", targets: ["Custom-CIFAR10"]),
        .executable(name: "ResNet-CIFAR10", targets: ["ResNet-CIFAR10"]),
        .executable(name: "LeNet-MNIST", targets: ["LeNet-MNIST"]),
        .executable(name: "MobileNet-Imagenette", targets: ["MobileNet-Imagenette"]),
        .executable(name: "MiniGoDemo", targets: ["MiniGoDemo"]),
        .executable(name: "Transformer", targets: ["Transformer"]),
        .library(name: "MiniGo", targets: ["MiniGo"]),
        .executable(name: "GAN", targets: ["GAN"]),
        .executable(name: "DCGAN", targets: ["DCGAN"]),
        .executable(name: "FastStyleTransferDemo", targets: ["FastStyleTransferDemo"]),
        .library(name: "FastStyleTransfer", targets: ["FastStyleTransfer"]),
        .executable(name: "Benchmarks", targets: ["Benchmarks"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.7.0"),
        .package(url: "https://github.com/kylef/Commander.git", from: "0.9.1"),
    ],
    targets: [
        .target(name: "Batcher", path: "Batcher"),
        .target(name: "ImageClassificationModels", path: "Models/ImageClassification"),
        .target(name: "Datasets", dependencies: ["ModelSupport", "Batcher"], path: "Datasets"),
        .target(name: "ModelSupport", dependencies: ["SwiftProtobuf"], path: "Support"),
        .target(name: "ImageClassificationModels", path: "Models/ImageClassification"),
        .target(name: "TextModels", dependencies: ["Datasets"], path: "Models/Text"),
        .target(
            name: "Autoencoder", dependencies: ["Datasets", "ModelSupport"], path: "Autoencoder"),
        .target(name: "Catch", path: "Catch"),
        .target(name: "Gym-FrozenLake", path: "Gym/FrozenLake"),
        .target(name: "Gym-CartPole", path: "Gym/CartPole"),
        .target(name: "Gym-Blackjack", path: "Gym/Blackjack"),
        .target(
            name: "VGG-Imagewoof", dependencies: ["ImageClassificationModels", "Datasets"],
            path: "Examples/VGG-Imagewoof"),
        .target(
            name: "Custom-CIFAR10", dependencies: ["Datasets"],
            path: "Examples/Custom-CIFAR10"),
        .target(
            name: "ResNet-CIFAR10", dependencies: ["ImageClassificationModels", "Datasets"],
            path: "Examples/ResNet-CIFAR10"),
        .target(
            name: "LeNet-MNIST", dependencies: ["ImageClassificationModels", "Datasets"],
            path: "Examples/LeNet-MNIST"),
        .target(
            name: "MobileNet-Imagenette", dependencies: ["ImageClassificationModels", "Datasets"],
            path: "Examples/MobileNet-Imagenette"),
        .target(
            name: "MiniGo", dependencies: ["ModelSupport"], path: "MiniGo", exclude: ["main.swift"]),
        .target(
            name: "MiniGoDemo", dependencies: ["MiniGo"], path: "MiniGo", sources: ["main.swift"]),
        .testTarget(name: "MiniGoTests", dependencies: ["MiniGo"]),
        .testTarget(name: "ImageClassificationTests", dependencies: ["ImageClassificationModels"]),
        .testTarget(name: "DatasetsTests", dependencies: ["Datasets"]),
        .target(name: "Transformer", dependencies: ["ModelSupport"], path: "Transformer"),
        .target(name: "GAN", dependencies: ["Datasets", "ModelSupport"], path: "GAN"),
        .target(name: "DCGAN", dependencies: ["Datasets", "ModelSupport"], path: "DCGAN"),
        .target(
            name: "FastStyleTransfer", dependencies: ["ModelSupport"], path: "FastStyleTransfer",
            exclude: ["Demo"]),
        .target(
            name: "FastStyleTransferDemo", dependencies: ["FastStyleTransfer"],
            path: "FastStyleTransfer/Demo"),
        .testTarget(name: "FastStyleTransferTests", dependencies: ["FastStyleTransfer"]),
        .target(
            name: "Benchmarks",
            dependencies: ["Datasets", "ModelSupport", "ImageClassificationModels", "Commander"],
            path: "Benchmarks"),
        .testTarget(name: "CheckpointTests", dependencies: ["ModelSupport"]),
    ]
)

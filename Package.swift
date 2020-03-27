// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TensorFlowModels",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library(name: "Batcher", targets: ["Batcher"]),
        .library(name: "Datasets", targets: ["Datasets"]),
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
        .library(name: "ImageClassificationModels", targets: ["ImageClassificationModels"]),
        .library(name: "RecommendationModels", targets: ["RecommendationModels"]),
        .library(name: "TextModels", targets: ["TextModels"]),
        .library(name: "TranslationModels", targets: ["TranslationModels"]),
        .executable(name: "Benchmarks", targets: ["Benchmarks"]),
        .executable(name: "VGG-Imagewoof", targets: ["VGG-Imagewoof"]),
        .executable(name: "Regression-BostonHousing", targets: ["Regression-BostonHousing"]),
        .executable(name: "Custom-CIFAR10", targets: ["Custom-CIFAR10"]),
        .executable(name: "ResNet-CIFAR10", targets: ["ResNet-CIFAR10"]),
        .executable(name: "LeNet-MNIST", targets: ["LeNet-MNIST"]),
        .executable(name: "MobileNet-Imagenette", targets: ["MobileNet-Imagenette"]),
        .executable(name: "GAN", targets: ["GAN"]),
        .executable(name: "DCGAN", targets: ["DCGAN"]),
        .executable(name: "BERT-CoLA", targets: ["BERT-CoLA"]),
        .library(name: "FastStyleTransfer", targets: ["FastStyleTransfer"]),
        .executable(name: "FastStyleTransferDemo", targets: ["FastStyleTransferDemo"]),
        .library(name: "MiniGo", targets: ["MiniGo"]),
        .executable(name: "MiniGoDemo", targets: ["MiniGoDemo"]),
        .executable(name: "GPT2-Inference", targets: ["GPT2-Inference"]),
        .executable(name: "GPT2-WikiText2", targets: ["GPT2-WikiText2"]),
        .executable(name: "Transformer-Translation", targets: ["Transformer-Translation"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.7.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMinor(from: "0.0.1")),
    ],
    targets: [
        .target(name: "Batcher", path: "Batcher"),
        .target(name: "Datasets", dependencies: ["ModelSupport", "Batcher"], path: "Datasets"),
        .target(name: "ModelSupport", dependencies: ["SwiftProtobuf"], path: "Support"),
        .target(name: "ImageClassificationModels", path: "Models/ImageClassification"),
        .target(name: "TextModels", dependencies: ["Datasets"], path: "Models/Text"),
        .target(name: "RecommendationModels", path: "Models/Recommendation"),
        .target(
            name: "Autoencoder1D", dependencies: ["Datasets", "ModelSupport"],
            path: "Autoencoder/Autoencoder1D"),
        .target(name: "TranslationModels", dependencies: ["Datasets"], path: "Models/Translation"),
        .target(
            name: "Autoencoder2D", dependencies: ["Datasets", "ModelSupport"],
            path: "Autoencoder/Autoencoder2D"),
        .target(name: "Catch", path: "Catch"),
        .target(name: "Gym-FrozenLake", path: "Gym/FrozenLake"),
        .target(name: "Gym-CartPole", path: "Gym/CartPole"),
        .target(name: "Gym-Blackjack", path: "Gym/Blackjack"),
        .target(
            name: "VGG-Imagewoof", dependencies: ["ImageClassificationModels", "Datasets"],
            path: "Examples/VGG-Imagewoof"),
        .target(
            name: "Regression-BostonHousing", dependencies: ["Datasets"],
            path: "Examples/Regression-BostonHousing"),
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
        .testTarget(name: "RecommendationModelTests", dependencies: ["RecommendationModels"]),
        .testTarget(name: "DatasetsTests", dependencies: ["Datasets", "TextModels"]),
        .target(
            name: "GPT2-Inference", dependencies: ["TextModels"],
            path: "Examples/GPT2-Inference",
            exclude: ["UI/Windows/main.swift", "UI/macOS/main.swift"]),
        .target(
            name: "GPT2-WikiText2",
            dependencies: ["Batcher", "Datasets", "TextModels"],
            path: "Examples/GPT2-WikiText2",
            exclude: ["UI/Windows/main.swift"]),
        .target(
            name: "Transformer-Translation",
            dependencies: ["TranslationModels", "Datasets", "ModelSupport"],
            path: "Examples/Transformer-Translation"
            ),
        .testTarget(name: "TextTests", dependencies: ["TextModels"]),
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
            dependencies: ["Datasets", "ModelSupport", "ImageClassificationModels", "ArgumentParser"],
            path: "Benchmarks"),
        .testTarget(name: "CheckpointTests", dependencies: ["ModelSupport"]),
        .target(
            name: "BERT-CoLA", dependencies: ["TextModels", "Datasets"], path: "Examples/BERT-CoLA"),
        .testTarget(name: "SupportTests", dependencies: ["ModelSupport"]),
    ]
)

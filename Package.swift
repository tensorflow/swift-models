// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-models",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library(name: "Checkpoints", targets: ["Checkpoints"]),
        .library(name: "Datasets", targets: ["Datasets"]),
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
        .library(name: "ImageClassificationModels", targets: ["ImageClassificationModels"]),
        .library(name: "VideoClassificationModels", targets: ["VideoClassificationModels"]),
        .library(name: "RecommendationModels", targets: ["RecommendationModels"]),
        .library(name: "TextModels", targets: ["TextModels"]),
        .library(name: "FastStyleTransfer", targets: ["FastStyleTransfer"]),
        .library(name: "MiniGo", targets: ["MiniGo"]),
        .library(name: "TrainingLoop", targets: ["TrainingLoop"]),
        .library(name: "BenchmarksCore", targets: ["BenchmarksCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.10.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMinor(from: "0.2.0")),
        .package(url: "https://github.com/google/swift-benchmark", .revision("f70bf472b00aeaa05e2374373568c2fe459c11c7")),
    ],
    targets: [
        .target(
            name: "Checkpoints", dependencies: ["SwiftProtobuf", "ModelSupport"],
            path: "Checkpoints"),
        .target(name: "Datasets", dependencies: ["ModelSupport"], path: "Datasets"),
        .target(name: "STBImage", path: "Support/STBImage"),
        .target(
            name: "ModelSupport", dependencies: ["SwiftProtobuf", "STBImage"], path: "Support",
            exclude: ["STBImage"]),
        .target(name: "ImageClassificationModels", path: "Models/ImageClassification"),
        .target(name: "VideoClassificationModels", path: "Models/Spatiotemporal"),
        .target(name: "TextModels", dependencies: ["Checkpoints", "Datasets"], path: "Models/Text"),
        .target(name: "RecommendationModels", path: "Models/Recommendation"),
        .target(name: "TrainingLoop", dependencies: ["ModelSupport"], path: "TrainingLoop"),
        .target(
            name: "Autoencoder1D", dependencies: ["Datasets", "ModelSupport", "TrainingLoop"],
            path: "Autoencoder/Autoencoder1D"),
        .target(
            name: "Autoencoder2D", dependencies: ["Datasets", "ModelSupport"],
            path: "Autoencoder/Autoencoder2D"),
        .target(
            name: "VariationalAutoencoder1D", dependencies: ["Datasets", "ModelSupport"],
            path: "Autoencoder/VAE1D"),
        .target(name: "Catch", path: "Catch"),
        .target(name: "Gym-FrozenLake", path: "Gym/FrozenLake"),
        .target(name: "Gym-CartPole", path: "Gym/CartPole"),
        .target(name: "Gym-Blackjack", path: "Gym/Blackjack"),
        .target(name: "Gym-DQN", path: "Gym/DQN"),
        .target(name: "Gym-PPO", path: "Gym/PPO"),
        .target(
            name: "VGG-Imagewoof",
            dependencies: ["Datasets", "ImageClassificationModels", "TrainingLoop"],
            path: "Examples/VGG-Imagewoof"),
        .target(
            name: "Regression-BostonHousing", dependencies: ["Datasets"],
            path: "Examples/Regression-BostonHousing"),
        .target(
            name: "Custom-CIFAR10", dependencies: ["Datasets"],
            path: "Examples/Custom-CIFAR10"),
        .target(
            name: "ResNet-CIFAR10",
            dependencies: ["Datasets", "ImageClassificationModels", "TrainingLoop"],
            path: "Examples/ResNet-CIFAR10"),
        .target(
            name: "Shallow-Water-PDE",
            dependencies: ["ArgumentParser", "Benchmark", "ModelSupport"],
            path: "Examples/Shallow-Water-PDE"),
        .target(
            name: "LeNet-MNIST",
            dependencies: ["Datasets", "ImageClassificationModels", "TrainingLoop"],
            path: "Examples/LeNet-MNIST"),
        .target(
            name: "MobileNetV1-Imagenette",
            dependencies: ["Datasets", "ImageClassificationModels", "TrainingLoop"],
            path: "Examples/MobileNetV1-Imagenette"),
        .target(
            name: "MobileNetV2-Imagenette",
            dependencies: ["Datasets", "ImageClassificationModels", "TrainingLoop"],
            path: "Examples/MobileNetV2-Imagenette"),
        .target(
            name: "PersonLab", dependencies: ["Checkpoints", "ModelSupport", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "PersonLab"),
        .target(
            name: "MiniGo", dependencies: ["Checkpoints"], path: "MiniGo", exclude: ["main.swift"]),
        .target(
            name: "MiniGoDemo", dependencies: ["MiniGo"], path: "MiniGo", sources: ["main.swift"]),
        .target(
            name: "NeuMF-MovieLens", dependencies: ["RecommendationModels", "Datasets"],
            path: "Examples/NeuMF-MovieLens"),
        .testTarget(name: "MiniGoTests", dependencies: ["MiniGo"]),
        .testTarget(name: "ImageClassificationTests", dependencies: ["ImageClassificationModels"]),
        .testTarget(name: "VideoClassificationTests", dependencies: ["VideoClassificationModels"]),
        .testTarget(name: "RecommendationModelTests", dependencies: ["RecommendationModels"]),
        .testTarget(name: "DatasetsTests", dependencies: ["Datasets", "TextModels"]),
        .target(
            name: "GPT2-Inference", dependencies: ["TextModels"],
            path: "Examples/GPT2-Inference",
            exclude: ["UI/Windows/main.swift", "UI/macOS/main.swift"]),
        .target(
            name: "GPT2-WikiText2",
            dependencies: ["Datasets", "TextModels", "TrainingLoop"],
            path: "Examples/GPT2-WikiText2",
            exclude: ["UI/Windows/main.swift"]),
        .testTarget(name: "TextTests", dependencies: ["TextModels"]),
        .target(name: "GAN", dependencies: ["Datasets", "ModelSupport"], path: "GAN"),
        .target(name: "DCGAN", dependencies: ["Datasets", "ModelSupport"], path: "DCGAN"),
        .target(
            name: "FastStyleTransfer", dependencies: ["Checkpoints"], path: "FastStyleTransfer",
            exclude: ["Demo"]),
        .target(
            name: "FastStyleTransferDemo", dependencies: ["FastStyleTransfer"],
            path: "FastStyleTransfer/Demo"),
        .testTarget(name: "FastStyleTransferTests", dependencies: ["FastStyleTransfer"]),
        .target(
            name: "BenchmarksCore",
            dependencies: [
                "Datasets", "ModelSupport", "ImageClassificationModels", "ArgumentParser",
                "TextModels", "Benchmark"
            ],
            path: "BenchmarksCore"),
        .target(name: "Benchmarks", dependencies: ["BenchmarksCore"], path: "Benchmarks"),
        .testTarget(name: "CheckpointTests", dependencies: ["Checkpoints"]),
        .target(
            name: "BERT-CoLA", dependencies: ["TextModels", "Datasets"], path: "Examples/BERT-CoLA"),
        .testTarget(name: "SupportTests", dependencies: ["ModelSupport"]),
        .target(
            name: "CycleGAN",
            dependencies: ["ArgumentParser", "ModelSupport", "Datasets"],
            path: "CycleGAN"
        ),
        .target(
            name: "pix2pix",
            dependencies: ["ArgumentParser", "ModelSupport", "Datasets"],
            path: "pix2pix"
        ),
        .target(
            name: "WordSeg",
            dependencies: ["ArgumentParser", "Datasets", "ModelSupport", "TextModels"],
            path: "Examples/WordSeg"
        ),
       .target(
           name: "Fractals",
           dependencies: ["ArgumentParser", "ModelSupport"],
           path: "Examples/Fractals"
       ),
       .target(
           name: "GrowingNeuralCellularAutomata",
           dependencies: ["ArgumentParser", "ModelSupport"],
           path: "Examples/GrowingNeuralCellularAutomata"
       )
    ]
)

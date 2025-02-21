// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-models",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .executable(name: "Benchmarks", targets: ["SwiftModelsBenchmarks"]),
        .library(name: "Checkpoints", targets: ["Checkpoints"]),
        .library(name: "Datasets", targets: ["Datasets"]),
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
        .library(name: "TensorBoard", targets: ["TensorBoard"]),
        .library(name: "ImageClassificationModels", targets: ["ImageClassificationModels"]),
        .library(name: "VideoClassificationModels", targets: ["VideoClassificationModels"]),
        .library(name: "RecommendationModels", targets: ["RecommendationModels"]),
        .library(name: "TextModels", targets: ["TextModels"]),
        .library(name: "FastStyleTransfer", targets: ["FastStyleTransfer"]),
        .library(name: "MiniGo", targets: ["MiniGo"]),
        .library(name: "TrainingLoop", targets: ["TrainingLoop"]),
        .library(name: "pix2pix", targets: ["pix2pix"]),
        .library(name: "SwiftModelsBenchmarksCore", targets: ["SwiftModelsBenchmarksCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.10.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", .branch("main")),
        .package(url: "https://github.com/google/swift-benchmark", from: "0.1.0"),
        .package(url: "https://github.com/tensorflow/swift-apis", .branch("main")),
    ],
    targets: [
        .target(
            name: "Checkpoints", dependencies: ["TensorFlow", "SwiftProtobuf", "ModelSupport"],
            path: "Checkpoints"),
        .target(name: "Datasets", dependencies: ["TensorFlow", "ModelSupport"], path: "Datasets"),
        .target(name: "STBImage", path: "Support/STBImage"),
        .target(
            name: "ModelSupport", dependencies: ["TensorFlow", "STBImage"], path: "Support",
            exclude: ["STBImage"]),
        .target(
            name: "TensorBoard",
            dependencies: ["TensorFlow", "SwiftProtobuf", "ModelSupport", "TrainingLoop"],
            path: "TensorBoard"),
        .target(
            name: "ImageClassificationModels", dependencies: ["TensorFlow"],
            path: "Models/ImageClassification"),
        .target(
            name: "VideoClassificationModels", dependencies: ["TensorFlow"],
            path: "Models/Spatiotemporal"
        ),
        .target(
            name: "TextModels",
            dependencies: ["Checkpoints", "Datasets", "SwiftProtobuf"],
            path: "Models/Text"),
        .target(
            name: "RecommendationModels", dependencies: ["TensorFlow"],
            path: "Models/Recommendation"),
        .target(name: "TrainingLoop", dependencies: ["ModelSupport"], path: "TrainingLoop"),
        .target(
            name: "Autoencoder1D",
            dependencies: ["Datasets", "ModelSupport", "TrainingLoop", "AutoencoderCallback"],
            path: "Autoencoder/Autoencoder1D"),
        .target(
            name: "Autoencoder2D", dependencies: ["Datasets", "ModelSupport"],
            path: "Autoencoder/Autoencoder2D"),
        .target(
            name: "VariationalAutoencoder1D", dependencies: ["Datasets", "ModelSupport"],
            path: "Autoencoder/VAE1D"),
        .target(
            name: "AutoencoderCallback", dependencies: ["ModelSupport", "TrainingLoop"],
            path: "Autoencoder/Callback"),
        .target(name: "Catch", dependencies: ["TensorFlow"], path: "Catch"),
        .target(name: "Gym-FrozenLake", dependencies: ["TensorFlow"], path: "Gym/FrozenLake"),
        .target(name: "Gym-CartPole", dependencies: ["TensorFlow"], path: "Gym/CartPole"),
        .target(name: "Gym-Blackjack", dependencies: ["TensorFlow"], path: "Gym/Blackjack"),
        .target(name: "Gym-DQN", dependencies: ["TensorFlow"], path: "Gym/DQN"),
        .target(name: "Gym-PPO", dependencies: ["TensorFlow"], path: "Gym/PPO"),
        .target(
            name: "VGG-Imagewoof",
            dependencies: ["TensorFlow", "Datasets", "ImageClassificationModels", "TrainingLoop"],
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
        .target(name: "BigTransfer-CIFAR100", 
            dependencies: ["Datasets", "ImageClassificationModels"], 
            path: "Examples/BigTransfer-CIFAR100"),
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
            name: "ResNet50-ImageNet",
            dependencies: ["Datasets", "ImageClassificationModels", "TrainingLoop", "TensorBoard"],
            path: "Examples/ResNet50-ImageNet"),
        .target(
            name: "PersonLab",
            dependencies: [
                "Checkpoints", "ModelSupport",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
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
            dependencies: ["Datasets", "TextModels", "TrainingLoop", "TensorBoard"],
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
            name: "SwiftModelsBenchmarksCore",
            dependencies: [
                "Datasets", "ModelSupport", "ImageClassificationModels", "ArgumentParser",
                "TextModels", "Benchmark",
            ],
            path: "SwiftModelsBenchmarksCore"),
        .target(
            name: "SwiftModelsBenchmarks",
            dependencies: ["SwiftModelsBenchmarksCore"],
            path: "SwiftModelsBenchmarks"
        ),
        .testTarget(name: "CheckpointTests", dependencies: ["Checkpoints", "ImageClassificationModels"]),
        .target(
            name: "BERT-CoLA",
            dependencies: ["x10_optimizers_optimizer", "TextModels", "Datasets", "TrainingLoop"],
            path: "Examples/BERT-CoLA"),
        .testTarget(name: "SupportTests", dependencies: ["ModelSupport"]),
        .target(
            name: "CycleGAN",
            dependencies: ["ArgumentParser", "ModelSupport", "Datasets"],
            path: "CycleGAN"
        ),
        .target(
            name: "pix2pix",
            dependencies: ["ArgumentParser", "ModelSupport", "Datasets", "Checkpoints"],
            path: "pix2pix",
            exclude: ["main.swift"]
        ),
        .target(
            name: "pix2pixDemo", dependencies: ["pix2pix"], path: "pix2pix", sources: ["main.swift"]),
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
        ),
    ]
)

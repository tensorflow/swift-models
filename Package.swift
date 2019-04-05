// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TensorFlowModels",
    products: [
        .executable(name: "MNIST", targets: ["MNIST"]),
        .executable(name: "CIFAR", targets: ["CIFAR"]),
        .executable(name: "MiniGo", targets: ["MiniGo", "MiniGoMain"]),
    ],
    targets: [
        .target(
            name: "MNIST",
            dependencies: [],
            path: "MNIST"),
        .target(
            name: "CIFAR",
            dependencies: [],
            path: "CIFAR"),
        .target(
            name: "MiniGoMain",
            dependencies: ["MiniGo"],
            path: "MiniGo",
            sources: ["main.swift"]),
        .target(
            name: "MiniGo",
            dependencies: [],
            path: "MiniGo",
            exclude: ["main.swift"]),
        .testTarget(
            name: "MiniGoTests",
            dependencies: ["MiniGo"]),
    ]
)

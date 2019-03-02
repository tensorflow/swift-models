// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MiniGo",
    targets: [
        .target(
            name: "MiniGo",
            dependencies: ["Game"],
            path: "Sources/MiniGo"),
        // TODO(xiejw): Refactor the subfolders to comply swift pm structure.
        .target(
            name: "Game",
            dependencies: [],
            path: "Sources",
            sources: [
                "GameLib",
                "Models",
                "Play",
                "Strategies",
            ]),
        .testTarget(
            name: "GameTests",
            dependencies: ["Game"]),
    ]
)

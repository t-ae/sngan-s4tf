// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SNGAN",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "CustomLayers",
            targets: ["CustomLayers"]),
    ],
    dependencies: [
        .package(url: "https://github.com/t-ae/tensorboardx-s4tf.git", from: "0.0.6"),
        .package(url: "https://github.com/t-ae/image-loader.git", from: "0.1.5"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(name: "CustomLayers"),
        .target(
            name: "SNGAN",
            dependencies: ["TensorBoardX", "ImageLoader", "CustomLayers"]),
        .testTarget(
            name: "CustomLayersTests",
            dependencies: ["CustomLayers"]),
        .testTarget(
            name: "SNGANTests",
            dependencies: ["SNGAN"]),
    ]
)

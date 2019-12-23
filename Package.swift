// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SNGAN",
    dependencies: [
        .package(url: "https://github.com/t-ae/tensorboardx-s4tf.git", from: "0.0.4"),
        .package(url: "https://github.com/t-ae/image-loader.git", from: "0.0.2"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "SNGAN",
            dependencies: ["TensorBoardX", "ImageLoader"]),
        .testTarget(
            name: "SNGANTests",
            dependencies: ["SNGAN"]),
    ]
)

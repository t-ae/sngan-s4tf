// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SNGAN",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
    ],
    dependencies: [
        .package(url: "https://github.com/t-ae/gan-utils-s4tf.git", from: "0.1.3"),
        .package(url: "https://github.com/t-ae/tensorboardx-s4tf.git", from: "0.0.11"),
        .package(url: "https://github.com/t-ae/image-loader.git", from: "0.1.8"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "SNGAN",
            dependencies: ["GANUtils", "TensorBoardX", "ImageLoader"]),
        .testTarget(
            name: "SNGANTests",
            dependencies: ["SNGAN"]),
    ]
)

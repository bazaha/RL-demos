// swift-tools-version: 5.9
import PackageDescription

// Engine-only package: rules + Core ML inference + MCTS, no UI.
// Drop it into an app target, or `swift build` it standalone on Apple Silicon.
let package = Package(
    name: "GomokuEngine",
    // iOS 18 / macOS 15 is the floor for MLMultiArray's typed Float16
    // accessors. Both target devices ship well past it; the exported
    // model itself only needs iOS 17.
    platforms: [.iOS("18.0"), .macOS("15.0")],
    products: [
        .library(name: "GomokuEngine", targets: ["GomokuEngine"]),
    ],
    targets: [
        .target(name: "GomokuEngine"),
        .testTarget(name: "GomokuEngineTests", dependencies: ["GomokuEngine"]),
    ]
)

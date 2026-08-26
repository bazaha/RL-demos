import Foundation

/// One reference position and how far this engine drifted from it.
public struct SelfTestResult: Sendable {
    public let name: String
    public let argmaxOK: Bool
    public let maxDeltaPolicy: Float
    public let deltaValue: Float
    public var passed: Bool {
        argmaxOK && maxDeltaPolicy <= EngineSelfTest.policyTolerance
            && deltaValue <= EngineSelfTest.valueTolerance
    }
}

/// Replays the shared reference positions through this engine.
///
/// `testvec.json` is produced by `scripts/export_gomoku_web.py` and copied next
/// to the Core ML model by `scripts/export_gomoku_coreml.py`. The browser build
/// runs the same five vectors at boot, and doing so caught a real numerical bug
/// there (single-pass GroupNorm variance blowing up on the empty board), so
/// keep this wired into app startup rather than treating it as a unit test.
///
/// It checks the *whole* path -- position replay, plane encoding, Core ML
/// inference, legal masking, softmax -- not just the model.
public enum EngineSelfTest {
    /// Same thresholds the play page applies to its WebGL2 engine.
    public static let policyTolerance: Float = 5e-3
    public static let valueTolerance: Float = 2e-2

    private struct Vector: Decodable {
        let name: String
        let moves: [Int]
        let policy: [Float]
        let value: Float
        let argmax: Int
    }

    private struct File: Decodable {
        let board: Int
        let vectors: [Vector]
    }

    public static func run(net: AZNet, testVectorURL url: URL) throws -> [SelfTestResult] {
        let file = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        precondition(file.board == net.config.board,
                     "testvec board \(file.board) != model board \(net.config.board)")
        return try file.vectors.map { vec in
            var state = GomokuState(config: net.config)
            for m in vec.moves { state.play(m) }
            let eval = try net.evaluate(state)
            var maxDelta: Float = 0
            for i in 0..<eval.policy.count {
                maxDelta = max(maxDelta, abs(eval.policy[i] - vec.policy[i]))
            }
            var argmax = 0
            for i in 1..<eval.policy.count where eval.policy[i] > eval.policy[argmax] {
                argmax = i
            }
            return SelfTestResult(name: vec.name, argmaxOK: argmax == vec.argmax,
                                  maxDeltaPolicy: maxDelta,
                                  deltaValue: abs(eval.value - vec.value))
        }
    }
}

import CoreML
import Foundation

public enum AZNetError: Error {
    case missingInput(String)
    case missingOutput(String)
    case batchOverflow(requested: Int, capacity: Int)
}

/// One network evaluation of a position.
public struct Evaluation: Sendable {
    /// Legal-masked softmax over all cells; illegal cells are exactly 0.
    public let policy: [Float]
    /// tanh value from the mover's point of view.
    public let value: Float
}

/// Core ML wrapper around the exported AlphaZero net.
///
/// Produced by `scripts/export_gomoku_coreml.py`, which also verifies that
/// every compute op prefers the ANE and that the model reproduces the shared
/// reference vectors (`testvec.json`).
///
/// Measured on an M2 Ultra, batch 1: **0.665 ms on the ANE vs 2.86 ms on the
/// GPU and 2.86 ms with `.all`** -- the planner does *not* pick the ANE on its
/// own, so `.cpuAndNeuralEngine` below is load-bearing, not a hint.
public final class AZNet {
    public let config: GomokuConfig
    /// Fixed batch dimension baked into this model file. Export more sizes with
    /// CML_BATCHES if you want to batch leaf evaluations.
    public let batchCapacity: Int

    private let model: MLModel
    private let input: MLMultiArray
    private let cellCount: Int

    public init(model: MLModel) throws {
        self.model = model
        let meta = (model.modelDescription.metadata[.creatorDefinedKey]
                    as? [String: String]) ?? [:]
        let board = meta["board"].flatMap(Int.init) ?? 15
        self.config = GomokuConfig(
            board: board,
            nInRow: meta["n_in_row"].flatMap(Int.init) ?? 5,
            cPuct: meta["c_puct"].flatMap(Float.init) ?? 3.0)
        self.cellCount = board * board

        guard let c = model.modelDescription
            .inputDescriptionsByName["x"]?.multiArrayConstraint else {
            throw AZNetError.missingInput("x")
        }
        self.batchCapacity = c.shape[0].intValue
        // allocated once and reused: 400 predictions per move should not mean
        // 400 allocations
        self.input = try MLMultiArray(shape: c.shape, dataType: .float16)
    }

    /// Loads a `.mlmodelc` directly, or compiles a `.mlpackage` first.
    /// Xcode precompiles models it bundles, so in an app you normally pass the
    /// `.mlmodelc` URL from `Bundle.main`.
    public static func load(
        url: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> AZNet {
        let compiled: URL = url.pathExtension == "mlmodelc"
            ? url
            : try await MLModel.compileModel(at: url)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        return try AZNet(model: try MLModel(contentsOf: compiled, configuration: cfg))
    }

    /// Evaluates up to `batchCapacity` positions in one prediction.
    ///
    /// Short batches are padded by repeating the last position; the padding
    /// outputs are dropped. Padding is free relative to a second prediction --
    /// on the ANE a batch of 8 costs 3.43 ms against 0.67 ms for a batch of 1,
    /// so per-position cost still falls.
    public func evaluate(_ states: [GomokuState]) throws -> [Evaluation] {
        guard !states.isEmpty else { return [] }
        guard states.count <= batchCapacity else {
            throw AZNetError.batchOverflow(requested: states.count,
                                           capacity: batchCapacity)
        }
        let n = cellCount
        input.withUnsafeMutableBufferPointer(ofType: Float16.self) { buf, _ in
            guard let base = buf.baseAddress else { return }
            for slot in 0..<batchCapacity {
                let src = states[min(slot, states.count - 1)]
                src.encode(into: base + slot * 4 * n)
            }
        }

        let provider = try MLDictionaryFeatureProvider(
            dictionary: ["x": MLFeatureValue(multiArray: input)])
        let out = try model.prediction(from: provider)
        guard let logits = out.featureValue(for: "policy_logits")?.multiArrayValue else {
            throw AZNetError.missingOutput("policy_logits")
        }
        guard let values = out.featureValue(for: "value")?.multiArrayValue else {
            throw AZNetError.missingOutput("value")
        }

        var results: [Evaluation] = []
        results.reserveCapacity(states.count)
        logits.withUnsafeBufferPointer(ofType: Float16.self) { lb in
            values.withUnsafeBufferPointer(ofType: Float16.self) { vb in
                for (i, state) in states.enumerated() {
                    results.append(Evaluation(
                        policy: Self.maskedSoftmax(logits: lb, offset: i * n,
                                                   state: state),
                        value: Float(vb[i])))
                }
            }
        }
        return results
    }

    public func evaluate(_ state: GomokuState) throws -> Evaluation {
        try evaluate([state])[0]
    }

    /// The model emits raw logits; masking has to happen here, before the
    /// softmax, or occupied cells keep a share of the probability mass.
    private static func maskedSoftmax(
        logits: UnsafeBufferPointer<Float16>, offset: Int, state: GomokuState
    ) -> [Float] {
        let n = state.cells.count
        var maxLogit = -Float.greatestFiniteMagnitude
        for i in 0..<n where state.cells[i] == 0 {
            maxLogit = max(maxLogit, Float(logits[offset + i]))
        }
        var p = [Float](repeating: 0, count: n)
        var sum: Float = 0
        for i in 0..<n where state.cells[i] == 0 {
            let e = expf(Float(logits[offset + i]) - maxLogit)
            p[i] = e
            sum += e
        }
        if sum > 1e-12 {
            for i in 0..<n { p[i] /= sum }
        } else {                                   // no legal move left
            let legal = state.legalActions
            for i in legal { p[i] = 1 / Float(legal.count) }
        }
        return p
    }
}

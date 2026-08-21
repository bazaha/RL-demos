import CoreML
import Foundation

/// CoreML wrapper around the exported net. Outputs the legal-masked softmax
/// policy and the tanh value, matching the trainer's forward exactly (fp16
/// compute noise <= ~2e-3, validated against the reference vectors).
final class Evaluator {
    private let model: MLModel
    private var inputBuf = [Float](repeating: 0, count: 4 * Rules.cells)
    let computeUnitsLabel: String

    init() throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all      // ANE first, GPU/CPU fallback
        guard let url = Bundle.main.url(forResource: "Gomoku15", withExtension: "mlmodelc") else {
            throw NSError(domain: "Gomoku", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Gomoku15.mlmodelc missing from bundle"])
        }
        model = try MLModel(contentsOf: url, configuration: cfg)
        computeUnitsLabel = "all(ANE 优先)"
    }

    func infer(_ pos: Position) throws -> (policy: [Float], value: Float) {
        let arr = try MLMultiArray(shape: [1, 4, 15, 15], dataType: .float32)
        pos.encode(into: &inputBuf)
        inputBuf.withUnsafeBufferPointer { src in
            arr.dataPointer.bindMemory(to: Float.self, capacity: src.count)
                .update(from: src.baseAddress!, count: src.count)
        }
        let out = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["board": MLFeatureValue(multiArray: arr)]))
        guard let lg = out.featureValue(for: "policy_logits")?.multiArrayValue,
              let vv = out.featureValue(for: "value")?.multiArrayValue else {
            throw NSError(domain: "Gomoku", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "model outputs missing"])
        }
        // fp16 mlprograms hand back Float16 arrays on device; binding those
        // bytes as Float32 silently yields garbage, so read by dtype
        let logits = Evaluator.floats(lg, count: Rules.cells)
        var maxLogit = -Float.infinity
        for i in 0..<Rules.cells where pos.board[i] == 0 {
            maxLogit = max(maxLogit, logits[i])
        }
        var p = [Float](repeating: 0, count: Rules.cells)
        var s: Float = 0
        for i in 0..<Rules.cells where pos.board[i] == 0 {
            let e = expf(logits[i] - maxLogit)
            p[i] = e
            s += e
        }
        if s > 0 { for i in 0..<Rules.cells { p[i] /= s } }
        let value = Evaluator.floats(vv, count: 1)[0]
        return (p, value)
    }

    private static func floats(_ arr: MLMultiArray, count: Int) -> [Float] {
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: p, count: count))
        case .float16:
            let p = arr.dataPointer.bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(p[$0]) }
        default:
            return (0..<count).map { arr[$0].floatValue }
        }
    }

    /// Startup self-test against the bundled reference vectors: the same
    /// numerics gate the web engine and the local inference service use.
    struct SelfTest {
        let ok: Bool
        let worstPolicyDelta: Float
        let detail: String
    }

    func selfTest() -> SelfTest {
        guard let url = Bundle.main.url(forResource: "testvec", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let vecs = root["vectors"] as? [[String: Any]] else {
            return SelfTest(ok: false, worstPolicyDelta: 1, detail: "testvec.json 缺失")
        }
        var worst: Float = 0
        for vec in vecs {
            guard let moves = vec["moves"] as? [Int],
                  let refPolicy = vec["policy"] as? [Double],
                  let refArgmax = vec["argmax"] as? Int else { continue }
            let pos = Position()
            for a in moves { pos.play(a) }
            guard let (p, _) = try? infer(pos) else {
                return SelfTest(ok: false, worstPolicyDelta: 1, detail: "前向失败")
            }
            var am = 0
            for i in 0..<Rules.cells where p[i] > p[am] { am = i }
            for i in 0..<Rules.cells {
                worst = max(worst, abs(p[i] - Float(refPolicy[i])))
            }
            if am != refArgmax {
                return SelfTest(ok: false, worstPolicyDelta: worst,
                                detail: "argmax 不一致 (\(vec["name"] ?? ""))")
            }
        }
        return SelfTest(ok: worst < 5e-3, worstPolicyDelta: worst,
                        detail: String(format: "maxΔ %.1e", worst))
    }
}

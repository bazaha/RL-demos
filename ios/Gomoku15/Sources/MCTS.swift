import Foundation

/// MCTS, a faithful port of the trainer's `Tree` (PUCT c=3, terminal values
/// from the mover's POV, sign flip per level in backup, argmax play).
final class MCTSNode {
    let position: Position
    var priors: [Float]?
    var visits: [Float]?
    var valueSum: [Float]?
    var children: [Int: MCTSNode] = [:]
    var expanded = false

    init(_ p: Position) { position = p }
}

final class MCTS {
    static let cPuct: Float = 3.0
    private(set) var root: MCTSNode

    init(_ position: Position) {
        root = MCTSNode(position.copy())
    }

    func advance(_ a: Int) {
        if let child = root.children[a] {
            root = child
        } else {
            let p = root.position.copy()
            p.play(a)
            root = MCTSNode(p)
        }
    }

    private func select() -> (path: [(MCTSNode, Int)], leaf: MCTSNode, terminal: Float?) {
        var node = root
        var path: [(MCTSNode, Int)] = []
        while true {
            let st = node.position
            if st.done { return (path, node, st.terminalValue()) }
            guard node.expanded, let n = node.visits, let w = node.valueSum,
                  let pr = node.priors else { return (path, node, nil) }
            var sumN: Float = 0
            for v in n { sumN += v }
            let k = MCTS.cPuct * (sumN + 1).squareRoot()
            var best = -Float.infinity
            var bestA = -1
            for i in 0..<Rules.cells where st.board[i] == 0 {
                let q = n[i] > 0 ? w[i] / n[i] : 0
                let score = q + k * pr[i] / (1 + n[i])
                if score > best { best = score; bestA = i }
            }
            path.append((node, bestA))
            if let child = node.children[bestA] {
                node = child
            } else {
                let np = st.copy()
                np.play(bestA)
                let child = MCTSNode(np)
                node.children[bestA] = child
                node = child
            }
        }
    }

    private func expand(_ node: MCTSNode, policy: [Float]) {
        var p = policy
        var s: Float = 0
        for i in 0..<Rules.cells {
            if node.position.board[i] != 0 { p[i] = 0 }
            s += p[i]
        }
        if s > 1e-12 {
            for i in 0..<Rules.cells { p[i] /= s }
        } else {
            var nLegal: Float = 0
            for i in 0..<Rules.cells where node.position.board[i] == 0 { nLegal += 1 }
            for i in 0..<Rules.cells { p[i] = node.position.board[i] == 0 ? 1 / nLegal : 0 }
        }
        node.priors = p
        node.visits = [Float](repeating: 0, count: Rules.cells)
        node.valueSum = [Float](repeating: 0, count: Rules.cells)
        node.expanded = true
    }

    private func backup(_ path: [(MCTSNode, Int)], leafValue: Float) {
        var v = leafValue
        for (node, a) in path.reversed() {
            v = -v
            node.visits![a] += 1
            node.valueSum![a] += v
        }
    }

    struct Result {
        let move: Int
        let visits: [Float]?
        /// Root mover's POV (backup flips per level, so root sums already are).
        let value: Float
    }

    /// sims == 0: raw policy argmax. Cancellation checked between sims.
    func run(sims: Int, evaluator: Evaluator,
             progress: ((Int, Int) -> Void)? = nil,
             isCancelled: (() -> Bool)? = nil) throws -> Result {
        if sims == 0 {
            let (p, v) = try evaluator.infer(root.position)
            if !root.expanded { expand(root, policy: p) }
            var a = -1
            var best = -Float.infinity
            for i in 0..<Rules.cells where root.position.board[i] == 0 {
                if p[i] > best { best = p[i]; a = i }
            }
            return Result(move: a, visits: nil, value: v)
        }
        if !root.expanded {
            let (p, _) = try evaluator.infer(root.position)
            expand(root, policy: p)
        }
        for s in 0..<sims {
            if isCancelled?() == true { break }
            try autoreleasepool {
                let (path, leaf, terminal) = select()
                if let tv = terminal {
                    backup(path, leafValue: tv)
                } else {
                    let (p, v) = try evaluator.infer(leaf.position)
                    expand(leaf, policy: p)
                    backup(path, leafValue: v)
                }
            }
            if s % 16 == 15 { progress?(s + 1, sims) }
        }
        let n = root.visits!
        var a = 0
        for i in 1..<Rules.cells where n[i] > n[a] { a = i }
        var q: Float = 0
        if n[a] > 0 { q = root.valueSum![a] / n[a] }
        return Result(move: a, visits: n, value: q)
    }
}

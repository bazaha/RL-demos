import Foundation

/// KataGo-style analysis read out of a searched tree. All conversions between
/// the mover's point of view (what every tree level stores) and display points
/// of view live here and nowhere else.
public struct MoveCandidate: Sendable, Identifiable {
    public let move: Int
    public let visits: Int
    /// Mean value of this move from the analysis root mover's POV.
    public let q: Float
    public let prior: Float
    /// Principal variation starting with `move`: the most-visited line, at
    /// most `pvPlies` moves. Shorter when the tree ends or the game does.
    public let pv: [Int]
    public var id: Int { move }

    /// Win probability for the side to move at the analysis root.
    public var winrateMover: Float { (q + 1) / 2 }
}

public struct PositionAnalysis: Sendable {
    public let candidates: [MoveCandidate]
    /// Side to move at the analyzed position (+1 black, -1 white).
    public let toPlay: Int8
    /// `moveCount` of the analyzed root; lets a UI drop stale results.
    public let ply: Int

    /// Best-move value converted to black's POV (for a trend chart).
    public var vBlackBest: Float? {
        candidates.first.map { toPlay == 1 ? $0.q : -$0.q }
    }

    /// Ghost-stone color for the k-th PV ply (0-based): movers alternate.
    public func pvColor(_ k: Int) -> Int8 {
        k % 2 == 0 ? toPlay : -toPlay
    }
}

extension Tree {
    /// Top-K root moves by visit count, with Q/prior and a greedy (argmax-N)
    /// principal variation. nil until the root has been searched.
    ///
    /// Lives in the package because the PV walk needs `Node.children`, which
    /// is deliberately not public.
    public func analysis(topK: Int = 5, pvPlies: Int = 3) -> PositionAnalysis? {
        let node = root
        guard node.isExpanded else { return nil }
        let n = node.visits
        var order: [Int] = []
        for i in 0..<n.count where n[i] > 0 { order.append(i) }
        guard !order.isEmpty else { return nil }
        order.sort { n[$0] == n[$1] ? $0 < $1 : n[$0] > n[$1] }
        let cands = order.prefix(topK).map { a in
            MoveCandidate(move: a, visits: Int(n[a]), q: node.q(a),
                          prior: node.prior[a], pv: pv(from: a, plies: pvPlies))
        }
        return PositionAnalysis(candidates: Array(cands),
                                toPlay: node.state.toPlay,
                                ply: node.state.moveCount)
    }

    /// Most-visited continuation below root child `a`, at most `plies` moves
    /// total (including `a`). Stops at unexpanded or terminal nodes.
    private func pv(from a: Int, plies: Int) -> [Int] {
        var line = [a]
        var node = root.children[a]
        while line.count < plies, let cur = node, !cur.state.isOver,
              cur.isExpanded {
            var best = -1
            for i in 0..<cur.visits.count where cur.visits[i] > 0 {
                if best < 0 || cur.visits[i] > cur.visits[best] { best = i }
            }
            guard best >= 0 else { break }
            line.append(best)
            node = cur.children[best]
        }
        return line
    }
}

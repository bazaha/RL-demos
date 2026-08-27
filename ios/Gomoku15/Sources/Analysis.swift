import Foundation

/// KataGo-style analysis extracted from a searched MCTS tree. All sign
/// conversions between the mover's POV (what the tree stores at each level)
/// and display POVs live in this file and nowhere else.
struct Candidate: Identifiable {
    let move: Int
    let visitsN: Int
    /// Mean value of this move from the analysis root mover's POV.
    let q: Float
    let prior: Float
    /// Principal variation starting with `move`, argmax-N walk, <= pvPlies.
    let pv: [Int]
    var id: Int { move }

    /// Win probability for the side to move at the analysis root.
    var winrateMover: Float { (q + 1) / 2 }
}

struct Analysis {
    let candidates: [Candidate]
    /// Side to move at the analyzed position (+1 black, -1 white).
    let toPlay: Int8
    /// moves.count when the analysis was taken; stale results are dropped.
    let ply: Int

    /// Best-move value converted to black's POV (for the trend chart).
    var vBlackBest: Float? {
        guard let q = candidates.first?.q else { return nil }
        return toPlay == 1 ? q : -q
    }

    /// Ghost-stone color for the k-th PV ply (0-based): movers alternate.
    func pvColor(_ k: Int) -> Int8 {
        k % 2 == 0 ? toPlay : -toPlay
    }
}

extension MCTS {
    /// Top-K root moves by visit count with Q/prior and a greedy
    /// (argmax-N) principal variation. nil until the root has been searched.
    func analysis(ply: Int, topK: Int = 5, pvPlies: Int = 3) -> Analysis? {
        guard root.expanded, let n = root.visits, let w = root.valueSum,
              let pr = root.priors else { return nil }
        var order: [Int] = []
        for i in 0..<Rules.cells where n[i] > 0 { order.append(i) }
        guard !order.isEmpty else { return nil }
        order.sort { n[$0] == n[$1] ? $0 < $1 : n[$0] > n[$1] }
        let cands = order.prefix(topK).map { a in
            Candidate(move: a, visitsN: Int(n[a]), q: w[a] / n[a],
                      prior: pr[a], pv: pv(from: a, plies: pvPlies))
        }
        return Analysis(candidates: Array(cands), toPlay: root.position.toPlay,
                        ply: ply)
    }

    /// Walk the most-visited line below root child `a`, at most `plies` moves
    /// total (including `a`). Stops at unexpanded or terminal nodes.
    private func pv(from a: Int, plies: Int) -> [Int] {
        var line = [a]
        var node = root.children[a]
        while line.count < plies, let cur = node, !cur.position.done,
              cur.expanded, let n = cur.visits {
            var best = -1
            for i in 0..<Rules.cells where n[i] > 0 {
                if best < 0 || n[i] > n[best] { best = i }
            }
            guard best >= 0 else { break }
            line.append(best)
            node = cur.children[best]
        }
        return line
    }
}

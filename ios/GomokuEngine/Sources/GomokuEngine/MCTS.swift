import Foundation

/// A search-tree node. Edge statistics live in the *parent* as flat arrays,
/// AlphaZero style, so `N[a]` / `W[a]` describe the edge to child `a`.
public final class Node {
    public let state: GomokuState
    public private(set) var prior: [Float] = []
    public internal(set) var visits: [Float] = []
    public internal(set) var totalValue: [Float] = []
    public private(set) var isExpanded = false
    var children: [Int: Node] = [:]

    init(state: GomokuState) { self.state = state }

    /// Masks the network prior to legal moves and renormalises. Equivalent to
    /// the trainer's `Tree.expand`: a softmax over every cell followed by
    /// mask-and-renormalise is the same thing as a softmax over legal cells.
    func expand(prior probs: [Float]) {
        let n = state.cells.count
        var p = [Float](repeating: 0, count: n)
        var sum: Float = 0
        for i in 0..<n where state.cells[i] == 0 {
            p[i] = probs[i]
            sum += probs[i]
        }
        if sum > 1e-12 {
            for i in 0..<n { p[i] /= sum }
        } else {
            let legal = state.legalActions
            for i in legal { p[i] = 1 / Float(legal.count) }
        }
        prior = p
        visits = [Float](repeating: 0, count: n)
        totalValue = [Float](repeating: 0, count: n)
        isExpanded = true
    }

    /// Mean action value of edge `a`, from the point of view of the player to
    /// move at *this* node.
    public func q(_ a: Int) -> Float {
        visits[a] > 0 ? totalValue[a] / visits[a] : 0
    }
}

/// PUCT search. A line-by-line port of the trainer's `Tree`; the three places
/// a port usually goes wrong are called out in comments below.
public final class Tree {
    public private(set) var root: Node
    public let cPuct: Float

    public init(state: GomokuState, cPuct: Float? = nil) {
        self.root = Node(state: state)
        self.cPuct = cPuct ?? state.config.cPuct
    }

    /// Walks down by PUCT until it reaches a terminal position or an unexpanded
    /// leaf. Returns the edges taken, the leaf, and -- for a terminal leaf --
    /// its value from that leaf's mover's point of view.
    func select() -> (path: [(node: Node, action: Int)], leaf: Node, terminal: Float?) {
        var node = root
        var path: [(node: Node, action: Int)] = []
        while true {
            if node.state.isOver {
                return (path, node, node.state.terminalValue)
            }
            if !node.isExpanded {
                return (path, node, nil)
            }
            var sumN: Float = 0
            for v in node.visits { sumN += v }
            let k = cPuct * (sumN + 1).squareRoot()
            var best = -Float.greatestFiniteMagnitude
            var action = -1
            let cells = node.state.cells
            for i in 0..<cells.count where cells[i] == 0 {
                let score = node.q(i) + k * node.prior[i] / (1 + node.visits[i])
                if score > best {
                    best = score
                    action = i
                }
            }
            guard action >= 0 else {           // no legal move: treat as terminal
                return (path, node, node.state.terminalValue)
            }
            path.append((node, action))
            if let child = node.children[action] {
                node = child
            } else {
                let child = Node(state: node.state.playing(action))
                node.children[action] = child
                node = child
            }
        }
    }

    /// `leafValue` is from the *leaf mover's* point of view, so the sign flips
    /// once per level on the way up. Flip first, then accumulate -- the edge at
    /// the bottom of the path belongs to the leaf's parent, who is the leaf
    /// mover's opponent.
    static func backup(_ path: [(node: Node, action: Int)], _ leafValue: Float) {
        var v = leafValue
        for (node, action) in path.reversed() {
            v = -v
            node.visits[action] += 1
            node.totalValue[action] += v
        }
    }

    /// Reuses the subtree after a move is played, so the work spent on the
    /// opponent's reply is not thrown away.
    public func advance(_ action: Int) {
        if let child = root.children[action] {
            root = child
        } else {
            root = Node(state: root.state.playing(action))
        }
    }

    /// Visit counts at the root. Already from the root player's point of view --
    /// `backup` did the sign flips, so **do not negate** `q(a)` when you read
    /// the root's evaluation out for a UI.
    public var rootVisits: [Float] { root.isExpanded ? root.visits : [] }
}

import Foundation

public struct SearchResult: Sendable {
    /// Chosen move, or -1 if the position is already over.
    public let action: Int
    /// Mean value of the chosen edge, from the point of view of the player who
    /// just moved. Already root-relative -- see `Tree.rootVisits`.
    public let value: Float
    /// Root visit counts, for a "what did the AI look at" overlay.
    public let visits: [Float]
    /// Simulations actually run (differs from the request only on cancellation).
    public let simulations: Int
    public let elapsed: TimeInterval
}

/// Deterministic RNG (SplitMix64) so a seeded game replays exactly. The trainer
/// pins its seed for the same reason.
public struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Owns the game and the search tree, and runs MCTS off the main actor.
///
/// An actor rather than a class: the Core ML model, the tree and the position
/// have to stay consistent, and the app should never block its UI on a search.
/// The simulation loop itself stays synchronous inside one actor hop -- 400
/// awaits per move would cost more than the search.
public actor AZPlayer {
    private let net: AZNet
    private var tree: Tree
    private var rng: any RandomNumberGenerator
    /// Bumped by every call that replaces or advances the tree.
    ///
    /// `think` suspends at `Task.yield()`, and an actor releases its lock at a
    /// suspension point -- so `play`/`rewind`/`reset` from another task can land
    /// in the middle of a live search and swap the tree out from under it. The
    /// search checks this across each yield and abandons the run rather than
    /// keep walking a tree that is no longer the one it started on.
    private var treeGeneration = 0

    public let config: GomokuConfig

    public init(net: AZNet, rng: any RandomNumberGenerator = SystemRandomNumberGenerator()) {
        self.net = net
        self.config = net.config
        self.tree = Tree(state: GomokuState(config: net.config))
        self.rng = rng
    }

    public var state: GomokuState { tree.root.state }

    public func reset(to state: GomokuState? = nil) {
        tree = Tree(state: state ?? GomokuState(config: config))
        treeGeneration &+= 1
    }

    /// Plays a move and keeps the matching subtree.
    public func play(_ action: Int) {
        tree.advance(action)
        treeGeneration &+= 1
    }

    /// Rebuilds the position from a move list. Use this for undo: the tree
    /// cannot walk backwards, so undo means replay.
    public func rewind(to moves: [Int]) {
        var s = GomokuState(config: config)
        for m in moves { s.play(m) }
        tree = Tree(state: s)
        treeGeneration &+= 1
    }

    /// Runs `simulations` PUCT simulations and picks a move.
    ///
    /// - `simulations: 0` skips search entirely and returns the argmax of the
    ///   raw network prior -- one forward pass, the cheapest playable level.
    /// - `temperature: 0` takes the most-visited move (deterministic: the same
    ///   position always yields the same move). Small values (~0.3) diversify
    ///   games without meaningfully weakening play; that is what the trainer
    ///   uses for its evaluation matches.
    /// - `yieldEvery` bounds how long the actor is held between cancellation
    ///   checks. On the ANE a simulation is well under a millisecond, so 32 is
    ///   already sub-frame.
    public func think(
        simulations: Int = 400,
        temperature: Float = 0,
        yieldEvery: Int = 32,
        progress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> SearchResult {
        let start = Date()
        let root = tree.root
        guard !root.state.isOver else {
            return SearchResult(action: -1, value: 0, visits: [], simulations: 0,
                                elapsed: 0)
        }

        // the root evaluation is needed either way; keep its value so the
        // zero-simulation level can still report the network's own read of the
        // position instead of a placeholder
        var rootValue: Float?
        if !root.isExpanded {
            let eval = try net.evaluate(root.state)
            root.expand(prior: eval.policy)
            rootValue = eval.value
        }

        if simulations <= 0 {
            var best = -Float.greatestFiniteMagnitude
            var action = -1
            for i in 0..<root.state.cells.count where root.state.cells[i] == 0 {
                if root.prior[i] > best { best = root.prior[i]; action = i }
            }
            // mover's point of view, the same convention as `Tree.root.q`
            let value = try rootValue ?? net.evaluate(root.state).value
            return SearchResult(action: action, value: value, visits: [],
                                simulations: 0,
                                elapsed: Date().timeIntervalSince(start))
        }

        var done = 0
        let generation = treeGeneration
        for i in 0..<simulations {
            let (path, leaf, terminal) = tree.select()
            if let tv = terminal {
                Tree.backup(path, tv)
            } else {
                let eval = try net.evaluate(leaf.state)
                leaf.expand(prior: eval.policy)
                Tree.backup(path, eval.value)
            }
            done = i + 1
            if done % yieldEvery == 0 {
                // check before reporting: a cancelled search's last act should
                // not be to enqueue a UI update that lands after the reset
                if Task.isCancelled { break }
                progress?(done, simulations)
                await Task.yield()
                // someone reset/advanced the tree across that suspension
                if treeGeneration != generation { break }
            }
        }
        guard treeGeneration == generation else {
            return SearchResult(action: -1, value: 0, visits: [],
                                simulations: done,
                                elapsed: Date().timeIntervalSince(start))
        }

        let action = selectMove(temperature: temperature)
        let value = action >= 0 && tree.root.visits.indices.contains(action)
            ? tree.root.q(action) : 0
        return SearchResult(action: action, value: value,
                            visits: tree.root.visits, simulations: done,
                            elapsed: Date().timeIntervalSince(start))
    }

    /// Visit-count move selection, matching the trainer's `AZPlayer.move_batch`:
    /// normalise by the max *before* raising to 1/temperature, or the power
    /// overflows for small temperatures.
    private func selectMove(temperature: Float) -> Int {
        let n = tree.root.visits
        guard !n.isEmpty else { return -1 }
        var maxN: Float = 0
        var total: Float = 0
        for v in n { maxN = max(maxN, v); total += v }
        guard total > 0 else { return -1 }

        if temperature <= 0 {
            var best: Float = -1
            var action = -1
            for i in 0..<n.count where n[i] > best { best = n[i]; action = i }
            return action
        }

        var w = [Float](repeating: 0, count: n.count)
        var sum: Float = 0
        let invT = 1 / temperature
        for i in 0..<n.count where n[i] > 0 {
            let x = powf(n[i] / maxN, invT)
            w[i] = x
            sum += x
        }
        guard sum > 0 else { return -1 }
        var r = Float.random(in: 0..<sum, using: &rng)
        for i in 0..<w.count where w[i] > 0 {
            r -= w[i]
            if r <= 0 { return i }
        }
        return w.firstIndex(where: { $0 > 0 }) ?? -1
    }
}

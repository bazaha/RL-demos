import CoreML
import XCTest
@testable import GomokuEngine

/// Rules and search tests run everywhere. The Core ML tests need an exported
/// model; point GOMOKU_MODEL at it to enable them:
///
///   GOMOKU_MODEL=$PWD/results/coreml_export/GomokuAZ_b1.mlpackage \
///   GOMOKU_TESTVEC=$PWD/results/coreml_export/testvec.json \
///   swift test --package-path ios/GomokuEngine
final class GomokuEngineTests: XCTestCase {
    let config = GomokuConfig(board: 15, nInRow: 5)

    private func at(_ r: Int, _ c: Int) -> Int { r * config.board + c }

    // MARK: rules

    func testHorizontalFiveWins() {
        var s = GomokuState(config: config)
        for c in 0..<4 {                       // black builds, white parks far away
            s.play(at(7, c))
            s.play(at(0, c))
        }
        XCTAssertFalse(s.isOver)
        s.play(at(7, 4))                       // black's fifth
        XCTAssertTrue(s.isOver)
        XCTAssertEqual(s.winner, 1)
    }

    func testDiagonalFiveWins() {
        var s = GomokuState(config: config)
        for i in 0..<4 {
            s.play(at(3 + i, 3 + i))
            s.play(at(0, i))
        }
        s.play(at(7, 7))
        XCTAssertTrue(s.isOver)
        XCTAssertEqual(s.winner, 1)
        XCTAssertEqual(s.winningLine(through: 7, 7, player: 1)?.count, 5)
    }

    func testFourIsNotAWin() {
        var s = GomokuState(config: config)
        for i in 0..<3 {
            s.play(at(7, i))
            s.play(at(0, i))
        }
        s.play(at(7, 3))
        XCTAssertFalse(s.isOver)
    }

    /// Terminal value is from the point of view of whoever must move *now*,
    /// so the side that just won sees -1 from the loser's turn.
    func testTerminalValueIsMoverRelative() {
        var s = GomokuState(config: config)
        for c in 0..<4 {
            s.play(at(7, c))
            s.play(at(0, c))
        }
        s.play(at(7, 4))
        XCTAssertEqual(s.winner, 1)
        XCTAssertEqual(s.toPlay, -1)           // white is "to move" at the terminal
        XCTAssertEqual(s.terminalValue, -1)    // and white has lost
    }

    // MARK: encoding

    func testEncodePlanes() {
        var s = GomokuState(config: config)
        s.play(at(7, 7))                       // black centre; white to move
        let n = config.cellCount
        var buf = [Float16](repeating: 9, count: 4 * n)
        buf.withUnsafeMutableBufferPointer { s.encode(into: $0.baseAddress!) }

        XCTAssertEqual(buf[at(7, 7)], 0)           // plane 0 = mover's (white) stones
        XCTAssertEqual(buf[n + at(7, 7)], 1)       // plane 1 = opponent (black)
        XCTAssertEqual(buf[2 * n + at(7, 7)], 1)   // plane 2 = last move
        XCTAssertEqual(buf[3 * n], 0)              // plane 3 = 0 because white moves

        var s0 = GomokuState(config: config)
        var buf0 = [Float16](repeating: 9, count: 4 * n)
        buf0.withUnsafeMutableBufferPointer { s0.encode(into: $0.baseAddress!) }
        XCTAssertEqual(buf0[3 * n], 1)             // black to move on an empty board
        XCTAssertEqual(buf0[2 * n], 0)             // no last move yet
        s0.play(at(0, 0))
    }

    // MARK: search

    /// backup flips the sign once per level, so a win for the leaf mover is a
    /// loss for the parent who moved into it.
    func testBackupAlternatesSign() {
        var s = GomokuState(config: config)
        s.play(at(7, 7))
        let tree = Tree(state: GomokuState(config: config))
        let uniform = [Float](repeating: 1.0 / Float(config.cellCount),
                              count: config.cellCount)
        tree.root.expand(prior: uniform)
        let (path, leaf, _) = tree.select()
        leaf.expand(prior: uniform)
        let (path2, _, _) = tree.select()
        XCTAssertEqual(path.count, 1)
        XCTAssertEqual(path2.count, 2)

        Tree.backup(path2, 1.0)                // leaf mover wins
        let (n0, a0) = (path2[0].node, path2[0].action)
        let (n1, a1) = (path2[1].node, path2[1].action)
        XCTAssertEqual(n1.totalValue[a1], -1)  // that leaf's parent: loss
        XCTAssertEqual(n0.totalValue[a0], 1)   // grandparent (root, same side): win
        XCTAssertEqual(n0.visits[a0], 1)
    }

    func testExpandMasksOccupiedCells() {
        var s = GomokuState(config: config)
        s.play(at(7, 7))
        let node = Node(state: s)
        node.expand(prior: [Float](repeating: 1.0 / Float(config.cellCount),
                                   count: config.cellCount))
        XCTAssertEqual(node.prior[at(7, 7)], 0)
        XCTAssertEqual(node.prior.reduce(0, +), 1, accuracy: 1e-4)
    }

    func testAdvanceReusesSubtree() {
        let tree = Tree(state: GomokuState(config: config))
        let uniform = [Float](repeating: 1.0 / Float(config.cellCount),
                              count: config.cellCount)
        tree.root.expand(prior: uniform)
        let (path, leaf, _) = tree.select()
        leaf.expand(prior: uniform)
        Tree.backup(path, 0.5)
        let action = path[0].action
        tree.advance(action)
        XCTAssertTrue(tree.root === leaf)      // same object, statistics kept
        XCTAssertTrue(tree.root.isExpanded)
    }

    // MARK: Core ML (opt-in)

    private func loadNet() async throws -> (AZNet, URL)? {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["GOMOKU_MODEL"] else { return nil }
        let vecPath = env["GOMOKU_TESTVEC"]
            ?? URL(fileURLWithPath: modelPath).deletingLastPathComponent()
                .appendingPathComponent("testvec.json").path
        let net = try await AZNet.load(url: URL(fileURLWithPath: modelPath))
        return (net, URL(fileURLWithPath: vecPath))
    }

    func testCoreMLMatchesTrainingReferences() async throws {
        guard let (net, vecURL) = try await loadNet() else {
            throw XCTSkip("set GOMOKU_MODEL to run the Core ML tests")
        }
        XCTAssertEqual(net.config.board, 15)
        XCTAssertEqual(net.config.nInRow, 5)
        let results = try EngineSelfTest.run(net: net, testVectorURL: vecURL)
        XCTAssertEqual(results.count, 5)
        for r in results {
            print("  \(r.name): argmax \(r.argmaxOK) policy \(r.maxDeltaPolicy) "
                  + "value \(r.deltaValue)")
            XCTAssertTrue(r.passed, "\(r.name) failed engine parity")
        }
    }

    func testSearchPicksTheWinningMove() async throws {
        guard let (net, _) = try await loadNet() else {
            throw XCTSkip("set GOMOKU_MODEL to run the Core ML tests")
        }
        // black has four in a row with both ends open; anything but completing
        // it is a blunder the search must not make
        var s = GomokuState(config: config)
        for c in 3..<7 {
            s.play(at(7, c))
            s.play(at(1, c))
        }
        XCTAssertEqual(s.toPlay, 1)
        let player = AZPlayer(net: net, rng: SeededRNG(seed: 42))
        await player.reset(to: s)
        let out = try await player.think(simulations: 64)
        XCTAssertTrue(out.action == at(7, 2) || out.action == at(7, 7),
                      "expected a winning completion, got \(out.action)")
        XCTAssertEqual(out.simulations, 64)
        print("  win-in-1: \(out.action) value \(out.value) in "
              + String(format: "%.0f ms", out.elapsed * 1000))
    }

    /// Not an assertion about speed, a place to read it off. Run it on the
    /// target device to replace the desktop estimates with real numbers.
    /// From the empty board nothing terminates early, so every simulation
    /// really does cost one network evaluation.
    func testSearchThroughput() async throws {
        guard let (net, _) = try await loadNet() else {
            throw XCTSkip("set GOMOKU_MODEL to run the Core ML tests")
        }
        let player = AZPlayer(net: net, rng: SeededRNG(seed: 1))
        _ = try await player.think(simulations: 16)      // warm up the ANE
        await player.reset()
        let sims = 400
        let out = try await player.think(simulations: sims)
        let msPerSim = out.elapsed * 1000 / Double(out.simulations)
        print(String(format: "  %d sims in %.0f ms = %.3f ms/sim (%.2f moves/s)",
                     out.simulations, out.elapsed * 1000, msPerSim,
                     1 / out.elapsed))
        XCTAssertEqual(out.simulations, sims)
        XCTAssertGreaterThan(out.visits.reduce(0, +), 0)
        // Not a benchmark, a floor. On the ANE this is ~0.75 ms/sim; the GPU
        // and CPU paths are ~2.9 ms/forward, so 2.0 separates them with room
        // to spare on a loaded machine and turns a silent 4x regression into a
        // failure. Skipped where there is no ANE (the simulator has none).
        if Self.hasNeuralEngine {
            XCTAssertLessThan(msPerSim, 2.0,
                              "\(msPerSim) ms/sim — inference is not on the ANE")
        }
    }

    static var hasNeuralEngine: Bool {
        MLModel.availableComputeDevices.contains {
            if case .neuralEngine = $0 { return true }
            return false
        }
    }

    /// The single highest-leverage assertion in this suite: `.all` looks like
    /// the obvious value and is what the app used to pass, but the Core ML
    /// planner then picks the GPU for this GroupNorm graph -- 2.94 ms/forward
    /// against 0.66 ms (results/coreml_export/coreml_report.json). Nothing else
    /// in the build would notice.
    func testDefaultLoadAsksForTheNeuralEngine() async throws {
        guard let (net, _) = try await loadNet() else {
            throw XCTSkip("set GOMOKU_MODEL to run the Core ML tests")
        }
        XCTAssertEqual(net.computeUnits, .cpuAndNeuralEngine)
    }

    /// And that asking for it actually buys something, so the constant above
    /// cannot be right while the deployment is wrong.
    func testNeuralEngineIsFasterThanCPUOnly() async throws {
        guard let (net, _) = try await loadNet(),
              let modelPath = ProcessInfo.processInfo.environment["GOMOKU_MODEL"] else {
            throw XCTSkip("set GOMOKU_MODEL to run the Core ML tests")
        }
        guard Self.hasNeuralEngine else { throw XCTSkip("no Neural Engine on this host") }
        let cpu = try await AZNet.load(url: URL(fileURLWithPath: modelPath),
                                       computeUnits: .cpuOnly)
        let state = GomokuState(config: net.config)
        func timeOf(_ n: AZNet) throws -> Double {
            for _ in 0..<10 { _ = try n.evaluate(state) }
            let t0 = Date()
            for _ in 0..<100 { _ = try n.evaluate(state) }
            return Date().timeIntervalSince(t0)
        }
        let ane = try timeOf(net), only = try timeOf(cpu)
        print(String(format: "  ANE %.3f ms/fwd vs CPU-only %.3f ms/fwd (%.2fx)",
                     ane * 10, only * 10, only / ane))
        // Measured 2.29x here. That is lower than the exporter's 4.06x because
        // this times the whole `evaluate` -- encode, predict, masked softmax --
        // not just the forward. A regression to .all/GPU collapses it to ~1.0x,
        // so 1.5 separates them without sitting on top of the real number.
        XCTAssertGreaterThan(only / ane, 1.5,
                             "the ANE path is not meaningfully faster — is it actually being used?")
    }

    /// The deleted app-side suite covered all four directions; the package only
    /// covered two, leaving vertical and anti-diagonal asserted nowhere.
    func testFiveWinsInEveryDirection() {
        for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
            var s = GomokuState()
            let r0 = 7 - 2 * dr, c0 = 7 - 2 * dc
            var filler = 0
            for k in 0..<5 {
                XCTAssertFalse(s.isOver, "dir \(dr),\(dc) ended early at \(k)")
                s.play((r0 + k * dr) * 15 + (c0 + k * dc))
                if k < 4 {                       // white filler, parked far away
                    s.play(14 * 15 + filler)
                    filler += 2
                }
            }
            XCTAssertTrue(s.isOver, "dir \(dr),\(dc)")
            XCTAssertEqual(s.winner, 1, "dir \(dr),\(dc)")
            let last = (r0 + 4 * dr, c0 + 4 * dc)
            XCTAssertEqual(s.winningLine(through: last.0, last.1, player: 1)?.count, 5,
                           "dir \(dr),\(dc)")
        }
    }

    func testSearchIsDeterministicAtZeroTemperature() async throws {
        guard let (net, _) = try await loadNet() else {
            throw XCTSkip("set GOMOKU_MODEL to run the Core ML tests")
        }
        let player = AZPlayer(net: net, rng: SeededRNG(seed: 7))
        let a = try await player.think(simulations: 32).action
        await player.reset()
        let b = try await player.think(simulations: 32).action
        XCTAssertEqual(a, b)
    }
}

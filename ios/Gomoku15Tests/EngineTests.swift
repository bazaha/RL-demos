import CoreML
import GomokuEngine
import XCTest
@testable import Gomoku15

/// Rules, MCTS and encode-plane semantics are covered by the engine package's
/// own suite (`swift test --package-path ios/GomokuEngine`). What is left for
/// the app target is everything the package cannot see: that the model is
/// actually in the bundle under the name the app asks for, that it answers to
/// the feature name the app sends, that its geometry matches the UI's, and that
/// it still reproduces the training-side reference vectors once compiled by
/// Xcode for this platform.
///
/// That last point is not redundant with the exporter's own acceptance run:
/// the exporter checks on macOS, these run wherever the tests run, which is how
/// a platform-specific readback bug would surface.
final class BundledModelTests: XCTestCase {

    private func loadNet() async throws -> AZNet {
        let url = try XCTUnwrap(
            Bundle.main.url(forResource: "GomokuAZ_b1", withExtension: "mlmodelc"),
            "GomokuAZ_b1.mlmodelc is not in the app bundle — run scripts/refresh_ios_model.sh")
        return try await AZNet.load(url: url)
    }

    func testModelGeometryMatchesTheUI() async throws {
        let net = try await loadNet()
        XCTAssertEqual(net.config.board, Rules.board)
        XCTAssertEqual(net.config.cellCount, Rules.cells)
    }

    /// The whole path: position replay -> plane encoding -> Core ML -> legal
    /// mask -> softmax, against the vectors the trainer and the web build share.
    func testBundledModelMatchesReferenceVectors() async throws {
        let net = try await loadNet()
        let url = try XCTUnwrap(Bundle.main.url(forResource: "testvec", withExtension: "json"))
        let results = try EngineSelfTest.run(net: net, testVectorURL: url)
        XCTAssertFalse(results.isEmpty, "no reference vectors were checked")
        for r in results {
            XCTAssertTrue(r.argmaxOK, "\(r.name): argmax differs")
            XCTAssertLessThanOrEqual(r.maxDeltaPolicy, EngineSelfTest.policyTolerance, r.name)
            XCTAssertLessThanOrEqual(r.deltaValue, EngineSelfTest.valueTolerance, r.name)
        }
    }

    /// Guards the bug this platform has hit before: an fp16 output tensor that
    /// reads back as all zeros leaves a uniform policy, which every other
    /// assertion here could still pass by luck.
    func testPolicyIsNotDegenerate() async throws {
        let net = try await loadNet()
        var state = GomokuState(config: Rules.config)
        for a in [7 * 15 + 7, 7 * 15 + 8, 8 * 15 + 8] { state.play(a) }
        let eval = try net.evaluate(state)
        let mass = eval.policy.reduce(0, +)
        XCTAssertEqual(mass, 1, accuracy: 1e-3, "policy is not a distribution")
        // An all-zero logits readback yields a uniform 1/222 = 0.0045 over the
        // legal cells; the committed reference maxima run 0.038-0.988, so this
        // floor sits an order of magnitude clear of both.
        XCTAssertGreaterThan(eval.policy.max() ?? 0, 10.0 / Float(Rules.cells),
                             "policy is flat — the model output read back as zeros")
        for i in 0..<Rules.cells where state.cells[i] != 0 {
            XCTAssertEqual(eval.policy[i], 0, "occupied cell \(i) kept probability mass")
        }
    }
}

final class SearchTests: XCTestCase {

    private func player(after moves: [Int]) async throws -> AZPlayer {
        let url = try XCTUnwrap(
            Bundle.main.url(forResource: "GomokuAZ_b1", withExtension: "mlmodelc"))
        let p = AZPlayer(net: try await AZNet.load(url: url))
        var s = GomokuState(config: Rules.config)
        for a in moves { s.play(a) }
        await p.reset(to: s)
        return p
    }

    func testSearchBlocksTheFour() async throws {
        // black four on row 7 (cols 5..8) with the LEFT end already blocked by
        // white: (7,9) is the only saving move. (An OPEN four would be lost
        // whatever white plays, so the search could legitimately pick anything.)
        let p = try await player(after: [7 * 15 + 5, 7 * 15 + 4, 7 * 15 + 6, 0 * 15 + 3,
                                         7 * 15 + 7, 0 * 15 + 5, 7 * 15 + 8])
        let r = try await p.think(simulations: 64)
        XCTAssertEqual(r.action, 7 * 15 + 9,
                       "got \(GameViewModel.coordName(r.action))")
    }

    func testSearchTakesTheWin() async throws {
        // black four with one open end: black must complete the five
        let p = try await player(after: [7 * 15 + 5, 0 * 15 + 1, 7 * 15 + 6, 0 * 15 + 3,
                                         7 * 15 + 7, 0 * 15 + 5, 7 * 15 + 8, 0 * 15 + 7])
        let r = try await p.think(simulations: 64)
        XCTAssertTrue([7 * 15 + 4, 7 * 15 + 9].contains(r.action),
                      "got \(GameViewModel.coordName(r.action))")
        XCTAssertGreaterThan(r.value, 0.8, "a won position must read as won")
    }

    /// Pins the property `GameViewModel` depends on: the search notices
    /// cancellation promptly. The bound is expressed against the number of
    /// simulations that had actually run when cancel() was called, so it is
    /// machine-independent — a wall-clock bound would mean something different
    /// on the simulator (~9 ms/sim) than on the ANE (~0.75 ms/sim).
    func testCancellationStopsTheSearchPromptly() async throws {
        let p = try await player(after: [7 * 15 + 7])
        let yieldEvery = 32
        let seen = Counter()
        let reached = expectation(description: "search under way")
        reached.assertForOverFulfill = false
        let task = Task {
            try await p.think(simulations: 1_000_000, yieldEvery: yieldEvery,
                              progress: { done, _ in
                                  seen.set(done)
                                  if done >= yieldEvery * 2 { reached.fulfill() }
                              })
        }
        await fulfillment(of: [reached], timeout: 60)
        let atCancel = seen.value
        task.cancel()
        let r = try await task.value
        XCTAssertGreaterThanOrEqual(r.simulations, yieldEvery)
        // one more yield window is all it may take to notice
        XCTAssertLessThanOrEqual(r.simulations, atCancel + 4 * yieldEvery,
                                 "cancellation took \(r.simulations - atCancel) extra simulations")
        XCTAssertGreaterThan(r.action, -1, "a cancelled search still returns its best move")
    }
}

/// The progress callback is `@Sendable`, so a captured `var` needs a lock.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var v = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return v }
    func set(_ n: Int) { lock.lock(); v = max(v, n); lock.unlock() }
}

/// The invariant that an actor-based engine is most likely to break: the tree
/// the search walks must always be the position the board shows. Nothing else
/// in the suite would notice them drifting apart.
@MainActor
final class ViewModelStateTests: XCTestCase {

    private func readyViewModel() async throws -> GameViewModel {
        let vm = GameViewModel()
        await vm.waitUntilReady()
        try XCTSkipUnless(vm.engineReady, "engine did not load: \(vm.engineBadge)")
        return vm
    }

    func testEngineTreeTracksTheBoardAcrossPlayUndoAndReset() async throws {
        let vm = try await readyViewModel()
        vm.level = .raw                       // one forward per move, keeps this quick

        vm.tap(7 * 15 + 7)
        var engine = await vm.engineBoard()
        XCTAssertEqual(engine, vm.board, "after the human move + AI reply")
        XCTAssertEqual(vm.moves.count, 2, "the AI should have answered")

        vm.tap(6 * 15 + 8)
        engine = await vm.engineBoard()
        XCTAssertEqual(engine, vm.board, "after a second exchange")

        vm.undo()
        engine = await vm.engineBoard()
        XCTAssertEqual(engine, vm.board, "after undo — a dropped rewind shows up here")

        vm.newGame()
        engine = await vm.engineBoard()
        XCTAssertEqual(engine, vm.board, "after a new game")
        XCTAssertEqual(vm.board.filter { $0 != 0 }.count, 0)
    }

    /// The case that actually exercises cancellation: `undo()` schedules a
    /// rewind, and the `tap()` on the next line cancels it before it has run,
    /// because both are synchronous on the main actor. If the cancelled task
    /// skips its rewind, the following `.advance` lands on the pre-undo tree
    /// and the engine is searching a board nobody can see. `GomokuState.play`
    /// has no legality check, so the corruption is silent.
    func testResetCancelledByAnImmediateMoveStillSyncsTheTree() async throws {
        let vm = try await readyViewModel()
        vm.level = .raw

        vm.tap(7 * 15 + 7)
        _ = await vm.engineBoard()
        vm.tap(6 * 15 + 8)
        _ = await vm.engineBoard()
        XCTAssertEqual(vm.moves.count, 4)

        vm.undo()                      // schedules .rewind(...)
        vm.tap(12 * 15 + 12)           // cancels it, then schedules .advance(...)
        let engine = await vm.engineBoard()
        XCTAssertEqual(engine, vm.board,
                       "the cancelled rewind was dropped — engine tree and board disagree")
    }

    /// The raw-policy tier still has to report the network's own read of the
    /// position; it regressed to "no evaluation shown" once already.
    func testRawPolicyLevelStillReportsAValue() async throws {
        let vm = try await readyViewModel()
        vm.level = .raw
        vm.tap(7 * 15 + 7)
        _ = await vm.engineBoard()
        XCTAssertNotNil(vm.valueBlack, "raw level lost its win-probability readout")
        XCTAssertNotNil(vm.lastMoveMs)
    }
}

/// Performance floors, measured on whatever this suite is running on.
///
/// The SwiftPM suite covers this on macOS, but it cannot run on an iOS device,
/// and the device is the only place the answer actually matters: the compute
/// plan, the ANE's presence and the planner's choice all differ from a Mac.
/// Run with:
///   xcodebuild -scheme Gomoku15 -destination 'id=<device>' \
///     -configuration Release GOMOKU_TEAM_ID=... ENABLE_TESTABILITY=YES test
final class DevicePerformanceTests: XCTestCase {

    private static var hasNeuralEngine: Bool {
        MLModel.availableComputeDevices.contains {
            if case .neuralEngine = $0 { return true }
            return false
        }
    }

    private func modelURL() throws -> URL {
        try XCTUnwrap(Bundle.main.url(forResource: "GomokuAZ_b1", withExtension: "mlmodelc"))
    }

    private func msPerForward(_ net: AZNet) throws -> Double {
        let state = GomokuState(config: Rules.config)
        for _ in 0..<20 { _ = try net.evaluate(state) }
        let t0 = Date()
        for _ in 0..<200 { _ = try net.evaluate(state) }
        return Date().timeIntervalSince(t0) * 1000 / 200
    }

    /// The measurement CLAUDE.md could only make on a Mac. On macOS `.all`
    /// makes the planner pick the GPU and costs 4x; whether iOS does the same
    /// was an open question, and this is where it gets answered.
    func testComputeUnitChoiceOnThisDevice() async throws {
        let url = try modelURL()
        let ane = try await AZNet.load(url: url, computeUnits: .cpuAndNeuralEngine)
        let all = try await AZNet.load(url: url, computeUnits: .all)
        let cpu = try await AZNet.load(url: url, computeUnits: .cpuOnly)
        let (a, l, c) = (try msPerForward(ane), try msPerForward(all), try msPerForward(cpu))
        print(String(format: "  [%@] ANE %.3f ms/fwd · .all %.3f · .cpuOnly %.3f  (ANE is %.2fx CPU, %.2fx .all)",
                     Self.hasNeuralEngine ? "has ANE" : "no ANE", a, l, c, c / a, l / a))
        XCTAssertEqual(ane.computeUnits, .cpuAndNeuralEngine)
        guard Self.hasNeuralEngine else { throw XCTSkip("no Neural Engine here") }
        XCTAssertGreaterThan(c / a, 1.5, "the ANE path is not meaningfully faster than CPU-only")
    }

    /// A per-move budget, not a benchmark. 400 simulations is the app's default
    /// strength; if a move takes longer than a second on the target hardware
    /// the app is not on the ANE any more.
    func testFourHundredSimulationMoveStaysUnderABudget() async throws {
        guard Self.hasNeuralEngine else { throw XCTSkip("no Neural Engine here") }
        let player = AZPlayer(net: try await AZNet.load(url: try modelURL()))
        _ = try await player.think(simulations: 16)          // warm the ANE
        await player.reset()
        let r = try await player.think(simulations: 400)
        let msPerSim = r.elapsed * 1000 / Double(r.simulations)
        print(String(format: "  400 sims in %.0f ms = %.3f ms/sim (%.2f moves/s)",
                     r.elapsed * 1000, msPerSim, 1 / r.elapsed))
        XCTAssertEqual(r.simulations, 400)
        XCTAssertLessThan(msPerSim, 2.0, "\(msPerSim) ms/sim — inference is not on the ANE")
    }
}

/// "AI 视角" regressions. The numbers here are taken from a real 25-ply game
/// replayed on an iPad Air M4, where the overlay drew nothing at all on 5 plies.
final class HeatOverlayTests: XCTestCase {

    /// ply 10 of that game: 568 visits on the move the AI played (now
    /// occupied), 1 visit on the best alternative. The old rule required
    /// >= max(2, 568 * 0.08) = 45.4 visits, so it painted nothing.
    func testDecisiveSearchStillShowsItsAlternatives() {
        var board = [Int8](repeating: 0, count: Rules.cells)
        let played = 7 * 15 + 7
        board[played] = 1
        var heat = [Float](repeating: 0, count: Rules.cells)
        heat[played] = 568
        heat[7 * 15 + 8] = 1
        heat[8 * 15 + 7] = 1

        let cells = heatCells(heat: heat, board: board)
        XCTAssertFalse(cells.isEmpty, "a decisive search must still show what else it looked at")
        XCTAssertFalse(cells.contains { $0.index == played }, "the played cell is occupied")
        XCTAssertEqual(cells.first?.weight, 1, "the best alternative gets full weight")
    }

    /// The over-correction: normalising over empty cells alone lit 198 cells
    /// at ply 24. A wall of blue is as uninformative as an empty board.
    func testFlatSearchIsCapped() {
        var board = [Int8](repeating: 0, count: Rules.cells)
        board[0] = 1
        var heat = [Float](repeating: 5, count: Rules.cells)
        heat[0] = 400
        let cells = heatCells(heat: heat, board: board)
        XCTAssertLessThanOrEqual(cells.count, 12)
        XCTAssertFalse(cells.isEmpty)
    }

    /// When every visit went to the played move there genuinely is nothing to
    /// show, and the button must be disabled rather than toggling to a no-op.
    func testNothingToShowIsReportedAsNothing() {
        var board = [Int8](repeating: 0, count: Rules.cells)
        let played = 7 * 15 + 7
        board[played] = 1
        var heat = [Float](repeating: 0, count: Rules.cells)
        heat[played] = 715
        XCTAssertTrue(heatCells(heat: heat, board: board).isEmpty)
    }

    /// Ordering is what makes the alpha ramp meaningful.
    func testCellsComeBackStrongestFirst() {
        let board = [Int8](repeating: 0, count: Rules.cells)
        var heat = [Float](repeating: 0, count: Rules.cells)
        heat[10] = 30; heat[20] = 90; heat[30] = 60
        let cells = heatCells(heat: heat, board: board)
        XCTAssertEqual(cells.map(\.index), [20, 30, 10])
        XCTAssertEqual(cells.first?.weight, 1)
    }
}

final class BookTests: XCTestCase {
    func testBookLoadsAndLinesAreLegal() throws {
        guard let book = OpeningBook.load() else {
            return XCTFail("bundled gomoku_book.json failed to load")
        }
        XCTAssertEqual(book.board, Rules.board)
        XCTAssertFalse(book.openings.isEmpty)
        for op in book.openings {
            XCTAssertGreaterThanOrEqual(op.line.count, op.ply_book)
            XCTAssertEqual(op.n, op.black_wins + op.white_wins + op.draws)
            var state = GomokuState(config: Rules.config)
            for a in op.line {
                XCTAssertTrue(state.isLegal(a), "\(op.id): illegal move \(a)")
                XCTAssertFalse(state.isOver, "\(op.id): line continues past game end")
                state.play(a)
            }
            if let v = op.v_black {
                XCTAssertEqual(v.count, op.line.count)
                XCTAssertTrue(v.allSatisfy { $0 >= -1.0 && $0 <= 1.0 })
            }
        }
    }

    @MainActor
    func testStartFromOpeningRebuildsPosition() throws {
        guard let book = OpeningBook.load(), let op = book.openings.first else {
            return XCTFail("no book")
        }
        let vm = GameViewModel()
        vm.startFromOpening(op.line, plies: op.ply_book)
        XCTAssertEqual(vm.moves, Array(op.line.prefix(op.ply_book)))
        let stones = vm.board.filter { $0 != 0 }.count
        XCTAssertEqual(stones, op.ply_book)
    }
}

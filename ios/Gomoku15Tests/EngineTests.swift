import XCTest
@testable import Gomoku15

final class EngineTests: XCTestCase {

    // --- rules -----------------------------------------------------------
    func testFiveDetectionAllDirections() {
        for (dr, dc) in Rules.dirs {
            let p = Position()
            let r0 = 7 - 2 * dr, c0 = 7 - 2 * dc
            var fillerCol = 0
            for k in 0..<5 {
                XCTAssertFalse(p.done)
                p.play((r0 + k * dr) * Rules.board + (c0 + k * dc))
                if k < 4 {                        // white filler far away
                    p.play(14 * Rules.board + fillerCol)
                    fillerCol += 2
                }
            }
            XCTAssertTrue(p.done)
            XCTAssertEqual(p.winner, 1, "dir \(dr),\(dc)")
        }
    }

    func testFourIsNotAWin() {
        let p = Position()
        for k in 0..<4 {
            p.play(7 * Rules.board + 3 + k)
            p.play(0 * Rules.board + k)
        }
        XCTAssertFalse(p.done)
    }

    // --- CoreML parity with the training-side reference vectors -----------
    private struct Vec: Decodable {
        let name: String
        let moves: [Int]
        let policy: [Float]
        let value: Float
        let argmax: Int
    }
    private struct TV: Decodable { let vectors: [Vec] }

    private func loadVectors() throws -> [Vec] {
        let url = Bundle(for: EngineTests.self)
            .url(forResource: "testvec", withExtension: "json")
            ?? Bundle.main.url(forResource: "testvec", withExtension: "json")!
        return try JSONDecoder().decode(TV.self, from: Data(contentsOf: url)).vectors
    }

    func testCoreMLMatchesReferenceVectors() throws {
        let ev = try Evaluator()
        for vec in try loadVectors() {
            let pos = Position()
            for a in vec.moves { pos.play(a) }
            let (p, v) = try ev.infer(pos)
            var am = 0
            for i in 0..<Rules.cells where p[i] > p[am] { am = i }
            XCTAssertEqual(am, vec.argmax, vec.name)
            var worst: Float = 0
            for i in 0..<Rules.cells { worst = max(worst, abs(p[i] - vec.policy[i])) }
            XCTAssertLessThan(worst, 5e-3, vec.name)
            XCTAssertLessThan(abs(v - vec.value), 2e-2, vec.name)
        }
    }

    // --- MCTS tactics ------------------------------------------------------
    func testMCTSBlocksTheFour() throws {
        // black four on row 7 (cols 5..8) with the LEFT end already blocked
        // by white: (7,9) is the only saving move, so the search must find
        // it. (An OPEN four would be lost whatever white plays -- a search
        // that sees every reply losing may legitimately pick anything.)
        let ev = try Evaluator()
        let pos = Position()
        for a in [7 * 15 + 5, 7 * 15 + 4, 7 * 15 + 6, 0 * 15 + 3,
                  7 * 15 + 7, 0 * 15 + 5, 7 * 15 + 8] {
            pos.play(a)
        }
        let tree = MCTS(pos)
        let r = try tree.run(sims: 64, evaluator: ev)
        XCTAssertEqual(r.move, 7 * 15 + 9,
                       "got \(GameViewModel.coordName(r.move))")
    }

    func testMCTSTakesTheWin() throws {
        // black four with one open end: MCTS as black must complete the five
        let ev = try Evaluator()
        let pos = Position()
        for a in [7 * 15 + 5, 0 * 15 + 1, 7 * 15 + 6, 0 * 15 + 3,
                  7 * 15 + 7, 0 * 15 + 5, 7 * 15 + 8, 0 * 15 + 7] {
            pos.play(a)
        }
        let tree = MCTS(pos)
        let r = try tree.run(sims: 64, evaluator: ev)
        XCTAssertTrue([7 * 15 + 4, 7 * 15 + 9].contains(r.move))
        XCTAssertGreaterThan(r.value, 0.8, "a won position must read as won")
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
            let pos = Position()
            for a in op.line {
                XCTAssertTrue(a >= 0 && a < Rules.cells && pos.board[a] == 0,
                              "\(op.id): illegal move \(a)")
                XCTAssertFalse(pos.done, "\(op.id): line continues past game end")
                pos.play(a)
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

final class AnalysisTests: XCTestCase {
    // black four, both ends open, black to move: winning moves (7,4)/(7,9)
    private func wonPosition() -> Position {
        let pos = Position()
        for a in [7 * 15 + 5, 0 * 15 + 1, 7 * 15 + 6, 0 * 15 + 3,
                  7 * 15 + 7, 0 * 15 + 5, 7 * 15 + 8, 0 * 15 + 7] {
            pos.play(a)
        }
        return pos
    }

    // black four with left end blocked, white to move: (7,9) only saves
    private func mustBlockPosition() -> Position {
        let pos = Position()
        for a in [7 * 15 + 5, 7 * 15 + 4, 7 * 15 + 6, 0 * 15 + 3,
                  7 * 15 + 7, 0 * 15 + 5, 7 * 15 + 8] {
            pos.play(a)
        }
        return pos
    }

    func testAnalysisFindsWinningMove() throws {
        let ev = try Evaluator()
        let tree = MCTS(wonPosition())
        _ = try tree.run(sims: 128, evaluator: ev)
        guard let an = tree.analysis(ply: 8) else { return XCTFail("no analysis") }
        XCTAssertEqual(an.toPlay, 1)
        let top = an.candidates[0]
        XCTAssertTrue([7 * 15 + 4, 7 * 15 + 9].contains(top.move),
                      "got \(GameViewModel.coordName(top.move))")
        XCTAssertGreaterThan(top.q, 0.8, "winning move must read as won")
        XCTAssertGreaterThan(top.winrateMover, 0.9)
        XCTAssertEqual(top.pv.first, top.move)
        XCTAssertEqual(top.pv.count, 1, "PV must stop at the terminal node")
        XCTAssertEqual(an.vBlackBest, top.q, "black to move: vBlack == q")
    }

    func testAnalysisBlocksFourAndPOVFlips() throws {
        let ev = try Evaluator()
        let tree = MCTS(mustBlockPosition())
        _ = try tree.run(sims: 128, evaluator: ev)
        guard let an = tree.analysis(ply: 7) else { return XCTFail("no analysis") }
        XCTAssertEqual(an.toPlay, -1, "white to move")
        XCTAssertEqual(an.candidates[0].move, 7 * 15 + 9)
        // white POV q -> black POV flips the sign
        XCTAssertEqual(an.vBlackBest, -an.candidates[0].q)
        // PV ghost colors alternate starting with the mover
        XCTAssertEqual(an.pvColor(0), -1)
        XCTAssertEqual(an.pvColor(1), 1)
    }

    func testPVsAreLegalAndBounded() throws {
        let ev = try Evaluator()
        let base = mustBlockPosition()
        let tree = MCTS(base)
        _ = try tree.run(sims: 128, evaluator: ev)
        guard let an = tree.analysis(ply: 7) else { return XCTFail("no analysis") }
        XCTAssertLessThanOrEqual(an.candidates.count, 5)
        var lastN = Int.max
        for c in an.candidates {
            XCTAssertLessThanOrEqual(c.visitsN, lastN, "sorted by visits")
            lastN = c.visitsN
            XCTAssertLessThanOrEqual(c.pv.count, 3)
            let p = base.copy()
            for a in c.pv {
                XCTAssertTrue(p.board[a] == 0 && !p.done,
                              "PV must replay legally")
                p.play(a)
            }
        }
    }

    @MainActor
    func testAnalysisModeEndToEnd() async throws {
        let vm = GameViewModel()
        for _ in 0..<100 where !vm.engineOK {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertTrue(vm.engineOK)
        vm.level = .s128
        vm.toggleAnalysis()
        for _ in 0..<200 where vm.analysis == nil {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        guard let an = vm.analysis else { return XCTFail("analysis never arrived") }
        XCTAssertEqual(an.ply, 0)
        XCTAssertEqual(an.toPlay, 1)
        XCTAssertFalse(vm.winrateHistory.isEmpty)
        // the best line auto-previews; a board tap PLAYS immediately (no
        // two-tap interception -- that trap ate moves and stalled the AI)
        let top = an.candidates[0].move
        XCTAssertEqual(vm.previewMove, top, "top PV auto-previewed")
        XCTAssertEqual(vm.previewPV?.first, top)
        vm.tap(top)
        XCTAssertEqual(vm.moves.first, top, "single tap must play")
        XCTAssertNil(vm.previewMove, "preview cleared on apply")
        XCTAssertTrue(vm.thinking, "AI turn must start after the human move")
        vm.newGame()
    }
}

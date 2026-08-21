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

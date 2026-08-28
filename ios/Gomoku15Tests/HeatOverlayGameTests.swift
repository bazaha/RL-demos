import CoreML
import GomokuEngine
import XCTest
@testable import Gomoku15

/// End-to-end guard for "AI 视角": replay a whole self-played game and check the
/// overlay at every ply.
///
/// The unit tests in `HeatOverlayTests` pin the rule against three positions
/// taken from one such game. This runs the real search instead, because the
/// failure was not a wrong constant -- it was that the shape of a *decisive*
/// search (nearly all visits on the move that then gets played and occupied)
/// only shows up once the engine is actually playing well, around ply 10.
///
/// ANE-gated: 25 plies x 400 simulations is ~6 s on device and ~90 s on the
/// simulator, which has no Neural Engine.
final class HeatOverlayGameTests: XCTestCase {

    private static var hasNeuralEngine: Bool {
        MLModel.availableComputeDevices.contains {
            if case .neuralEngine = $0 { return true }
            return false
        }
    }

    func testOverlayIsNeverEmptyWhileTheSearchHasAnAlternative() async throws {
        guard Self.hasNeuralEngine else { throw XCTSkip("no Neural Engine here") }
        let url = try XCTUnwrap(Bundle.main.url(forResource: "GomokuAZ_b1",
                                                withExtension: "mlmodelc"))
        let player = AZPlayer(net: try await AZNet.load(url: url))
        var state = GomokuState(config: Rules.config)
        var pliesWithAlternatives = 0

        for ply in 1...40 {
            let r = try await player.think(simulations: 400)
            guard r.action >= 0 else { break }
            let heat = r.visits
            state.play(r.action)
            await player.play(r.action)
            let board = state.cells          // what the UI shows when it draws

            // did the search look anywhere other than the move it played?
            var bestAlternative: Float = 0
            for i in 0..<Rules.cells where board[i] == 0 {
                bestAlternative = max(bestAlternative, heat[i])
            }
            let cells = heatCells(heat: heat, board: board)
            XCTAssertLessThanOrEqual(cells.count, 12, "ply \(ply): overlay is a wall of blue")
            if bestAlternative > 0 {
                pliesWithAlternatives += 1
                XCTAssertFalse(cells.isEmpty,
                               "ply \(ply): search had an alternative worth \(bestAlternative) "
                               + "visits but the overlay drew nothing")
                XCTAssertEqual(try XCTUnwrap(cells.first).weight, 1, accuracy: 1e-6,
                               "ply \(ply): the best alternative should be at full strength")
            } else {
                XCTAssertTrue(cells.isEmpty, "ply \(ply): nothing was explored, nothing should draw")
            }
            if state.isOver { break }
        }

        XCTAssertGreaterThan(pliesWithAlternatives, 10,
                             "the game ended too early to exercise the decisive-search case")
    }
}

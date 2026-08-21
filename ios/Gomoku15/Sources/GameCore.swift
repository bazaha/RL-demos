import Foundation

/// Gomoku rules, a faithful port of the trainer's `State`
/// (train_rl_gomoku_alphazero.py). Anything that differs here breaks parity
/// with the reference vectors, so keep the semantics byte-for-byte.
enum Rules {
    static let board = 15
    static let cells = 225
    static let inRow = 5
    static let dirs: [(Int, Int)] = [(0, 1), (1, 0), (1, 1), (1, -1)]
}

final class Position {
    var board = [Int8](repeating: 0, count: Rules.cells)
    var toPlay: Int8 = 1          // +1 black, -1 white
    var lastMove: Int = -1
    var nMoves = 0
    var winner: Int8 = 0
    var done = false

    init() {}

    func copy() -> Position {
        let p = Position()
        p.board = board
        p.toPlay = toPlay
        p.lastMove = lastMove
        p.nMoves = nMoves
        p.winner = winner
        p.done = done
        return p
    }

    func play(_ a: Int) {
        let p = toPlay
        board[a] = p
        lastMove = a
        nMoves += 1
        if Position.winLine(board: board, at: a, player: p) != nil {
            winner = p
            done = true
        } else if nMoves == Rules.cells {
            winner = 0
            done = true
        }
        toPlay = -p
    }

    /// The >= 5 line through `at`, as cell indices, or nil. Used both for the
    /// win check and for highlighting the winning line in the UI.
    static func winLine(board: [Int8], at a: Int, player p: Int8) -> [Int]? {
        let B = Rules.board
        let r0 = a / B, c0 = a % B
        for (dr, dc) in Rules.dirs {
            var cells = [a]
            for sgn in [1, -1] {
                var r = r0 + sgn * dr, c = c0 + sgn * dc
                while r >= 0 && r < B && c >= 0 && c < B && board[r * B + c] == p {
                    if sgn == 1 { cells.append(r * B + c) } else { cells.insert(r * B + c, at: 0) }
                    r += sgn * dr
                    c += sgn * dc
                }
            }
            if cells.count >= Rules.inRow { return cells }
        }
        return nil
    }

    /// From the mover's point of view, exactly like the trainer.
    func terminalValue() -> Float {
        if winner == 0 { return 0 }
        return winner == toPlay ? 1 : -1
    }

    /// 4 planes, mover's POV, into a 4*225 float buffer (NCHW order).
    func encode(into buf: inout [Float]) {
        let n = Rules.cells
        for i in 0..<(4 * n) { buf[i] = 0 }
        let tp = toPlay
        for i in 0..<n {
            let v = board[i]
            if v == tp { buf[i] = 1 } else if v == -tp { buf[n + i] = 1 }
        }
        if lastMove >= 0 { buf[2 * n + lastMove] = 1 }
        if tp == 1 { for i in (3 * n)..<(4 * n) { buf[i] = 1 } }
    }
}

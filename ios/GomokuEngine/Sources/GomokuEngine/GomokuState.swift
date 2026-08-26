import Foundation

/// Board geometry. Read it off the Core ML model rather than hardcoding 15:
/// `AZNet` publishes the values the exporter stamped into the model metadata.
public struct GomokuConfig: Sendable, Equatable {
    public let board: Int
    public let nInRow: Int
    /// PUCT exploration constant. The trainer's default is 3.0 and the exporter
    /// copies whatever the training run actually used into the model metadata.
    public let cPuct: Float

    public var cellCount: Int { board * board }

    public init(board: Int = 15, nInRow: Int = 5, cPuct: Float = 3.0) {
        self.board = board
        self.nInRow = nInRow
        self.cPuct = cPuct
    }
}

/// A Gomoku position. Port of the trainer's `State`
/// (scripts/train_rl_gomoku_alphazero.py) -- keep the two in step, an on-device
/// engine that disagrees with training is worse than no engine.
///
/// A struct, so the trainer's explicit `clone()` is just a copy here.
public struct GomokuState: Sendable {
    public let config: GomokuConfig
    /// Row-major, +1 black / -1 white / 0 empty.
    public private(set) var cells: [Int8]
    /// +1 = black to move, -1 = white.
    public private(set) var toPlay: Int8
    public private(set) var lastMove: Int
    public private(set) var moveCount: Int
    public private(set) var winner: Int8
    public private(set) var isOver: Bool

    public init(config: GomokuConfig = GomokuConfig()) {
        self.config = config
        self.cells = [Int8](repeating: 0, count: config.cellCount)
        self.toPlay = 1
        self.lastMove = -1
        self.moveCount = 0
        self.winner = 0
        self.isOver = false
    }

    public func isLegal(_ action: Int) -> Bool {
        action >= 0 && action < cells.count && cells[action] == 0 && !isOver
    }

    public var legalActions: [Int] {
        isOver ? [] : (0..<cells.count).filter { cells[$0] == 0 }
    }

    public mutating func play(_ action: Int) {
        let b = config.board
        let r = action / b, c = action % b
        let p = toPlay
        cells[action] = p
        lastMove = action
        moveCount += 1
        if winningLine(through: r, c, player: p) != nil {
            winner = p
            isOver = true
        } else if moveCount == cells.count {
            winner = 0
            isOver = true
        }
        toPlay = -p
    }

    public func playing(_ action: Int) -> GomokuState {
        var s = self
        s.play(action)
        return s
    }

    /// Result from the point of view of whoever must move *now* -- the same
    /// convention the trainer uses, and the one MCTS backup depends on.
    public var terminalValue: Float {
        guard winner != 0 else { return 0 }
        return winner == toPlay ? 1 : -1
    }

    /// The >= nInRow line through (r, c), or nil. Also useful for highlighting
    /// the winning row in a UI.
    public func winningLine(through r: Int, _ c: Int, player: Int8) -> [Int]? {
        let b = config.board
        let dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for (dr, dc) in dirs {
            var line = [r * b + c]
            for sign in [1, -1] {
                var rr = r + sign * dr, cc = c + sign * dc
                while rr >= 0, rr < b, cc >= 0, cc < b, cells[rr * b + cc] == player {
                    if sign == 1 { line.append(rr * b + cc) } else { line.insert(rr * b + cc, at: 0) }
                    rr += sign * dr
                    cc += sign * dc
                }
            }
            if line.count >= config.nInRow { return line }
        }
        return nil
    }

    /// 4 planes of board x board, always from the mover's point of view:
    /// 0 = my stones, 1 = opponent stones, 2 = one-hot last move,
    /// 3 = all ones iff black is to move.
    ///
    /// Writes straight into `dst` (length 4 * cellCount) so the caller can hand
    /// it an MLMultiArray buffer without an intermediate allocation.
    public func encode(into dst: UnsafeMutablePointer<Float16>) {
        let n = cells.count
        for i in 0..<(4 * n) { dst[i] = 0 }
        let tp = toPlay
        for i in 0..<n {
            let v = cells[i]
            if v == tp { dst[i] = 1 } else if v == -tp { dst[n + i] = 1 }
        }
        if lastMove >= 0 { dst[2 * n + lastMove] = 1 }
        if tp == 1 { for i in 0..<n { dst[3 * n + i] = 1 } }
    }
}

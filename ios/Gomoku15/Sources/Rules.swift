import GomokuEngine

/// Board geometry for the UI layer only -- drawing, coordinate names, the
/// opening book.
///
/// The *engine* never reads this: `AZNet` publishes the real geometry from the
/// metadata the exporter stamped into the model, and `GameViewModel.boot()`
/// refuses to start when the two disagree. Keeping a constant here rather than
/// threading a config through every view is safe only because of that check --
/// do not remove it.
enum Rules {
    static let config = GomokuConfig()
    static let board = config.board
    static let cells = config.cellCount
}

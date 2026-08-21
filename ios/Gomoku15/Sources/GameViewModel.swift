import Foundation
import SwiftUI

@MainActor
final class GameViewModel: ObservableObject {
    enum Level: Int, CaseIterable, Identifiable {
        case raw = 0, s128 = 128, s400 = 400, s1600 = 1600
        var id: Int { rawValue }
        var label: String {
            switch self {
            case .raw: return "原始策略"
            case .s128: return "搜索 128"
            case .s400: return "搜索 400"
            case .s1600: return "搜索 1600"
            }
        }
    }

    @Published var board = [Int8](repeating: 0, count: Rules.cells)
    @Published var moves: [Int] = []
    @Published var humanSide: Int8 = 1
    @Published var level: Level = .s400
    @Published var thinking = false
    @Published var progressText = ""
    @Published var status = "加载引擎中…"
    @Published var statusIsGood: Bool? = nil
    @Published var winCells: [Int]? = nil
    @Published var heat: [Float]? = nil
    @Published var showHeat = false
    @Published var valueBlack: Float? = nil     // black's win prob source
    @Published var engineBadge = "引擎校验中…"
    @Published var engineOK = false
    @Published var lastMoveMs: Int? = nil
    @Published var gameOver = false

    private var position = Position()
    private var tree: MCTS!
    private var evaluator: Evaluator?
    private var aiTask: Task<Void, Never>? = nil

    init() {
        Task { await boot() }
    }

    private func boot() async {
        do {
            let ev = try Evaluator()
            let test = await Task.detached(priority: .userInitiated) { ev.selfTest() }.value
            evaluator = ev
            engineOK = test.ok
            engineBadge = test.ok ? "引擎校验 ✓ 与训练端一致（\(test.detail)）"
                                  : "引擎校验失败：\(test.detail)"
            newGame()
            if ProcessInfo.processInfo.arguments.contains("-autoplay") {
                await autoplayForScreenshots()
            }
        } catch {
            engineBadge = "引擎加载失败：\(error.localizedDescription)"
            status = "引擎不可用"
        }
    }

    func newGame() {
        aiTask?.cancel()
        position = Position()
        tree = MCTS(position)
        board = position.board
        moves = []
        winCells = nil
        heat = nil
        showHeat = false
        valueBlack = nil
        lastMoveMs = nil
        gameOver = false
        thinking = false
        statusIsGood = nil
        if position.toPlay != humanSide {
            status = "AI 开局中…"
            scheduleAITurn()
        } else {
            status = "轮到你落子。"
        }
    }

    func setSide(_ s: Int8) {
        guard !thinking, moves.isEmpty || gameOver else { return }
        humanSide = s
        newGame()
    }

    func tap(_ a: Int) {
        guard !thinking, !gameOver, position.toPlay == humanSide,
              a >= 0, a < Rules.cells, position.board[a] == 0 else { return }
        heat = nil
        showHeat = false
        apply(a)
        if !gameOver { scheduleAITurn() }
    }

    func undo() {
        guard !thinking, !moves.isEmpty else { return }
        var k = moves.count
        if position.toPlay == humanSide || gameOver { k -= 1 }
        k -= 1
        let keep = Array(moves.prefix(max(0, k)))
        position = Position()
        for a in keep { position.play(a) }
        tree = MCTS(position)
        moves = keep
        board = position.board
        winCells = nil
        heat = nil
        showHeat = false
        valueBlack = nil
        gameOver = false
        statusIsGood = nil
        if position.toPlay != humanSide {
            status = "已悔棋。"
            scheduleAITurn()
        } else {
            status = "已悔棋，轮到你。"
        }
    }

    private func apply(_ a: Int) {
        position.play(a)
        tree.advance(a)
        moves.append(a)
        board = position.board
        if position.done {
            gameOver = true
            if position.winner != 0 {
                winCells = Position.winLine(board: position.board, at: a,
                                            player: position.winner)
                let youWin = position.winner == humanSide
                status = youWin ? "你赢了！" : "AI 获胜。"
                statusIsGood = youWin
            } else {
                status = "平局（棋盘下满）。"
            }
            UINotificationFeedbackGenerator().notificationOccurred(
                position.winner == humanSide ? .success : .warning)
        } else {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
        }
    }

    private func scheduleAITurn() {
        guard let ev = evaluator, !gameOver,
              position.toPlay != humanSide, !thinking else { return }
        thinking = true
        let sims = level.rawValue
        status = sims > 0 ? "AI 思考中（\(sims) 次模拟）…" : "AI 思考中…"
        progressText = ""
        let searchTree = tree!
        aiTask = Task { [weak self] in
            let t0 = Date()
            let result: MCTS.Result?
            do {
                result = try await Task.detached(priority: .userInitiated) { () -> MCTS.Result in
                    try searchTree.run(sims: sims, evaluator: ev, progress: { done, total in
                        Task { @MainActor [weak self] in
                            self?.progressText = "\(done)/\(total)"
                        }
                    }, isCancelled: { Task.isCancelled })
                }.value
            } catch {
                result = nil
            }
            guard let self, !Task.isCancelled else { return }
            self.thinking = false
            self.progressText = ""
            guard let r = result, r.move >= 0 else {
                self.status = "引擎异常，请开新对局。"
                return
            }
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            self.lastMoveMs = ms
            self.heat = r.visits
            let aiIsBlack = self.position.toPlay == 1
            self.valueBlack = aiIsBlack ? r.value : -r.value
            self.apply(r.move)
            if !self.gameOver {
                self.status = "AI 落子 \(Self.coordName(r.move))（\(String(format: "%.1f", Double(ms) / 1000))s）。轮到你。"
            }
        }
    }

    /// Screenshot/UI-test hook: play a short scripted game at raw level.
    private func autoplayForScreenshots() async {
        level = .s128
        let human = [7 * 15 + 7, 6 * 15 + 8, 8 * 15 + 6]
        for a in human {
            while thinking { try? await Task.sleep(nanoseconds: 100_000_000) }
            if gameOver { break }
            tap(a)
        }
        while thinking { try? await Task.sleep(nanoseconds: 100_000_000) }
        if heat != nil { showHeat = true }
    }

    nonisolated static func coordName(_ a: Int) -> String {
        let cols = Array("ABCDEFGHJKLMNOP")
        let r = a / Rules.board, c = a % Rules.board
        return "\(cols[c])\(Rules.board - r)"
    }
}

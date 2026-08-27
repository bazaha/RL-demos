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

    // KataGo-style analysis mode
    @Published var analysisOn = false
    @Published var analysis: Analysis? = nil
    @Published var analyzing = false
    @Published var previewMove: Int? = nil          // selected candidate (1st tap)
    @Published var winrateHistory: [Int: Float] = [:]  // ply -> vBlack in [-1, 1]

    private var position = Position()
    private var tree: MCTS!
    private var evaluator: Evaluator?
    private var aiTask: Task<Void, Never>? = nil
    private var analysisTask: Task<Void, Never>? = nil
    private final class CancelFlag: @unchecked Sendable { var on = false }
    private var analysisFlag: CancelFlag? = nil

    /// PV of the selected candidate, for the board's ghost-stone preview.
    var previewPV: [Int]? {
        guard let a = previewMove else { return nil }
        return analysis?.candidates.first(where: { $0.move == a })?.pv
    }

    private var analysisSims: Int { level == .raw ? 400 : level.rawValue }

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
            if ProcessInfo.processInfo.arguments.contains("-analysis") {
                analysisOn = true
                await autoplayForScreenshots()
            } else if ProcessInfo.processInfo.arguments.contains("-autoplay") {
                await autoplayForScreenshots()
            }
        } catch {
            engineBadge = "引擎加载失败：\(error.localizedDescription)"
            status = "引擎不可用"
        }
    }

    func newGame() {
        aiTask?.cancel()
        cancelAnalysis()
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
        analysis = nil
        previewMove = nil
        winrateHistory = [:]
        if position.toPlay != humanSide {
            status = "AI 开局中…"
            scheduleAITurn()
        } else {
            status = "轮到你落子。"
            scheduleAnalysis()
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
        // Analysis mode: first tap on a marked candidate previews its PV,
        // a second tap on the same cell plays it (touch stand-in for hover).
        if analysisOn, previewMove != a,
           analysis?.candidates.contains(where: { $0.move == a }) == true {
            previewMove = a
            return
        }
        playHuman(a)
    }

    private func playHuman(_ a: Int) {
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
        reset(to: Array(moves.prefix(max(0, k))), statusPrefix: "已悔棋")
    }

    /// Load an opening-book line and continue play from there.
    func startFromOpening(_ line: [Int], plies: Int? = nil) {
        guard !thinking else { return }
        let k = min(plies ?? line.count, line.count)
        reset(to: Array(line.prefix(k)), statusPrefix: "已摆上开局前 \(k) 手")
    }

    // MARK: - Analysis mode (KataGo-style)

    func toggleAnalysis() {
        analysisOn.toggle()
        if analysisOn {
            scheduleAnalysis()
        } else {
            cancelAnalysis()
            analysis = nil
            previewMove = nil
            // winrateHistory kept: toggling back on continues the chart
        }
    }

    /// Table-row tap: toggle the PV preview for a marked candidate.
    func selectCandidate(_ a: Int) {
        guard analysis?.candidates.contains(where: { $0.move == a }) == true
        else { return }
        previewMove = previewMove == a ? nil : a
    }

    private func cancelAnalysis() {
        analysisFlag?.on = true
        analysisFlag = nil
        analysisTask?.cancel()
        analysisTask = nil
        analyzing = false
    }

    /// Analyze the current position (human to move) on an isolated tree.
    /// The game tree is never touched, so there is nothing to race with.
    private func scheduleAnalysis() {
        guard analysisOn, !gameOver, !thinking,
              position.toPlay == humanSide, let ev = evaluator else { return }
        cancelAnalysis()
        let ply = moves.count
        let sims = analysisSims
        let t = MCTS(position)       // init copies the position
        let flag = CancelFlag()
        analysisFlag = flag
        analyzing = true
        analysisTask = Task { [weak self] in
            let result: Analysis?
            do {
                _ = try await Task.detached(priority: .utility) { () -> MCTS.Result in
                    try t.run(sims: sims, evaluator: ev,
                              isCancelled: { flag.on })
                }.value
                result = flag.on ? nil : t.analysis(ply: ply)
            } catch {
                result = nil
            }
            guard let self, !Task.isCancelled else { return }
            self.analyzing = false
            guard let r = result, r.ply == self.moves.count,
                  !self.gameOver else { return }
            self.analysis = r
            if let v = r.vBlackBest { self.winrateHistory[ply] = v }
        }
    }

    /// Rebuild the game at a move prefix (shared by undo and the book) and
    /// hand the turn to whoever is due -- including triggering the AI.
    private func reset(to keep: [Int], statusPrefix: String) {
        aiTask?.cancel()
        cancelAnalysis()
        position = Position()
        for a in keep { position.play(a) }
        tree = MCTS(position)
        moves = keep
        board = position.board
        winCells = nil
        heat = nil
        showHeat = false
        valueBlack = nil
        gameOver = position.done
        thinking = false
        statusIsGood = nil
        analysis = nil
        previewMove = nil
        winrateHistory = winrateHistory.filter { $0.key <= keep.count }
        if !gameOver, position.toPlay != humanSide {
            status = "\(statusPrefix)。"
            scheduleAITurn()
        } else {
            status = "\(statusPrefix)，轮到你。"
            scheduleAnalysis()
        }
    }

    private func apply(_ a: Int) {
        cancelAnalysis()
        analysis = nil
        previewMove = nil
        position.play(a)
        tree.advance(a)
        moves.append(a)
        board = position.board
        if position.done {
            gameOver = true
            if analysisOn {
                winrateHistory[moves.count] =
                    position.winner == 0 ? 0 : (position.winner == 1 ? 1 : -1)
            }
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
            if self.analysisOn, let v = self.valueBlack {
                // the AI's own search doubles as this ply's analysis
                self.winrateHistory[self.moves.count] = v
            }
            self.apply(r.move)
            if !self.gameOver {
                self.status = "AI 落子 \(Self.coordName(r.move))（\(String(format: "%.1f", Double(ms) / 1000))s）。轮到你。"
                self.scheduleAnalysis()
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
            playHuman(a)    // bypass the two-tap candidate preview
        }
        while thinking { try? await Task.sleep(nanoseconds: 100_000_000) }
        if analysisOn {
            // let the post-move analysis land, then preview the top PV
            for _ in 0..<100 where analysis == nil {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            if let top = analysis?.candidates.first?.move { selectCandidate(top) }
        } else if heat != nil {
            showHeat = true
        }
    }

    nonisolated static func coordName(_ a: Int) -> String {
        let cols = Array("ABCDEFGHJKLMNOP")
        let r = a / Rules.board, c = a % Rules.board
        return "\(cols[c])\(Rules.board - r)"
    }
}

import Foundation
import GomokuEngine
import SwiftUI

/// Name of the Core ML model in the bundle. Matches what
/// `scripts/export_gomoku_coreml.py` writes, so the exporter's output can be
/// copied in verbatim -- see `scripts/refresh_ios_model.sh`.
private let modelResourceName = "GomokuAZ_b1"

enum EngineError: LocalizedError {
    case modelMissing(String)
    case testVectorsMissing
    case geometryMismatch(model: Int, ui: Int)

    var errorDescription: String? {
        switch self {
        case .modelMissing(let name):
            return "\(name).mlmodelc 不在 App 包里（跑 scripts/refresh_ios_model.sh 重新导出并放入）"
        case .testVectorsMissing:
            return "testvec.json 不在 App 包里"
        case .geometryMismatch(let model, let ui):
            return "模型棋盘 \(model)×\(model) 与界面的 \(ui)×\(ui) 不一致"
        }
    }
}

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

    /// Geometry actually in force. Replaced in `boot()` with the config the
    /// exporter stamped into the model, so the UI position and the engine's
    /// tree are built from one source rather than two that happen to agree.
    private var config = Rules.config
    /// UI-side authoritative position. The engine keeps its own copy inside the
    /// search tree; `runEngine` is the only thing that reconciles the two.
    private var state = GomokuState(config: Rules.config)
    private var player: AZPlayer?
    /// Exactly one engine task exists at a time. A new one cancels the previous
    /// and waits for it to unwind before touching the actor.
    private var engineTask: Task<Void, Never>?
    /// Bumped by every `runEngine`. A task only writes published state while it
    /// is still the current one, so a cancelled search cannot clear `thinking`
    /// out from under its successor or stamp stale text over a fresh status.
    private var engineGeneration = 0

    var engineReady: Bool { player != nil }

    /// Whether "AI 视角" has anything to paint. The button used to be enabled
    /// whenever `heat != nil`, which included every position where the search
    /// put all of its visits on the move it then played -- toggling it did
    /// nothing and looked broken.
    var hasHeatToShow: Bool {
        guard let heat else { return false }
        return !heatCells(heat: heat, board: board).isEmpty
    }

    /// Held so tests can await startup; `init` cannot be async.
    private var bootTask: Task<Void, Never>?

    init() {
        bootTask = Task { await boot() }
    }

    /// Test hook: resolves once the engine has loaded (or failed to).
    func waitUntilReady() async { await bootTask?.value }

    /// Test hook: waits for any in-flight engine work, then reports the board
    /// the search tree actually holds. It must always equal `board`; the whole
    /// risk of moving the engine into an actor is that these two drift.
    func engineBoard() async -> [Int8]? {
        await engineTask?.value
        guard let player else { return nil }
        return await player.state.cells
    }

    // MARK: - boot

    /// Loads the model and runs the shared reference vectors through the whole
    /// path (replay, encode, Core ML, mask, softmax) before play is allowed.
    ///
    /// `nonisolated` and `async`, so the ~0.9 s model load and the five warm-up
    /// forwards happen off the main actor -- doing this inline would block the
    /// first frame.
    private nonisolated static func bootEngine() async throws
        -> (player: AZPlayer, config: GomokuConfig, checks: [SelfTestResult]) {
        guard let modelURL = Bundle.main.url(forResource: modelResourceName,
                                             withExtension: "mlmodelc") else {
            throw EngineError.modelMissing(modelResourceName)
        }
        // .cpuAndNeuralEngine, not .all: with .all the Core ML planner picks the
        // GPU for this GroupNorm graph (macOS measurement in
        // results/coreml_export/coreml_report.json: 2.94 ms vs 0.66 ms).
        let net = try await AZNet.load(url: modelURL)
        guard net.config.board == Rules.board else {
            throw EngineError.geometryMismatch(model: net.config.board, ui: Rules.board)
        }
        guard let vecURL = Bundle.main.url(forResource: "testvec",
                                           withExtension: "json") else {
            throw EngineError.testVectorsMissing
        }
        let checks = try EngineSelfTest.run(net: net, testVectorURL: vecURL)
        return (AZPlayer(net: net), net.config, checks)
    }

    private func boot() async {
        do {
            let (player, config, checks) = try await Self.bootEngine()
            self.player = player
            self.config = config
            self.state = GomokuState(config: config)
            // an empty vector list is a failure, not a pass: the old self-test
            // skipped unparseable vectors and then reported maxD 0.0
            let worst = checks.map(\.maxDeltaPolicy).max() ?? 0
            let worstValue = checks.map(\.deltaValue).max() ?? 0
            engineOK = !checks.isEmpty && checks.allSatisfy(\.passed)
            if engineOK {
                engineBadge = String(format: "引擎校验 ✓ 与训练端一致（%d 个参考局面，policy maxΔ %.1e，value maxΔ %.1e）",
                                     checks.count, worst, worstValue)
            } else if checks.isEmpty {
                engineBadge = "引擎校验失败：参考局面为空"
            } else {
                let bad = checks.filter { !$0.passed }.map(\.name).joined(separator: ", ")
                engineBadge = String(format: "引擎校验失败：%@（policy maxΔ %.1e，value maxΔ %.1e）",
                                     bad, worst, worstValue)
            }
            newGame()
            if ProcessInfo.processInfo.arguments.contains("-autoplay") {
                await autoplayForScreenshots()
            }
        } catch {
            engineBadge = "引擎加载失败：\(error.localizedDescription)"
            status = "引擎不可用"
        }
    }

    // MARK: - game flow

    func newGame() {
        reset(to: [], playingStatus: "AI 开局中…", waitingStatus: "轮到你落子。")
    }

    func setSide(_ s: Int8) {
        guard !thinking, moves.isEmpty || gameOver else { return }
        humanSide = s
        newGame()
    }

    func tap(_ a: Int) {
        guard engineReady, !thinking, !gameOver,
              state.toPlay == humanSide, state.isLegal(a) else { return }
        heat = nil
        showHeat = false
        apply(a)
        // .advance keeps the subtree the last search already built
        runEngine(sync: .advance(a))
    }

    func undo() {
        guard !thinking, !moves.isEmpty else { return }
        var k = moves.count
        if state.toPlay == humanSide || gameOver { k -= 1 }
        k -= 1
        reset(to: Array(moves.prefix(max(0, k))), statusPrefix: "已悔棋")
    }

    /// Load an opening-book line and continue play from there.
    func startFromOpening(_ line: [Int], plies: Int? = nil) {
        guard !thinking else { return }
        let k = min(plies ?? line.count, line.count)
        reset(to: Array(line.prefix(k)), statusPrefix: "已摆上开局前 \(k) 手")
    }

    private func reset(to keep: [Int], statusPrefix: String) {
        reset(to: keep,
              playingStatus: "\(statusPrefix)。",
              waitingStatus: "\(statusPrefix)，轮到你。")
    }

    /// Rebuilds the game at a move prefix (new game, undo, opening book) and
    /// hands the turn to whoever is due -- including triggering the AI.
    private func reset(to keep: [Int], playingStatus: String, waitingStatus: String) {
        state = GomokuState(config: config)
        for a in keep { state.play(a) }
        moves = keep
        board = state.cells
        winCells = nil
        heat = nil
        showHeat = false
        valueBlack = nil
        lastMoveMs = nil
        gameOver = state.isOver
        statusIsGood = nil
        guard engineReady else {
            status = "引擎不可用"
            return
        }
        status = (!gameOver && state.toPlay != humanSide) ? playingStatus : waitingStatus
        // the tree cannot walk backwards, so a prefix change means replay
        runEngine(sync: .rewind(keep))
    }

    private func apply(_ a: Int) {
        let mover = state.toPlay
        let r = a / config.board, c = a % config.board
        state.play(a)
        moves.append(a)
        board = state.cells
        if state.isOver {
            gameOver = true
            if state.winner != 0 {
                winCells = state.winningLine(through: r, c, player: mover)
                let youWin = state.winner == humanSide
                status = youWin ? "你赢了！" : "AI 获胜。"
                statusIsGood = youWin
            } else {
                status = "平局（棋盘下满）。"
            }
            UINotificationFeedbackGenerator().notificationOccurred(
                state.winner == humanSide ? .success : .warning)
        } else {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
        }
    }

    // MARK: - engine

    private enum EngineSync {
        /// One move onto the existing tree; the matching subtree is kept.
        case advance(Int)
        /// Rebuild from a move list (new game, undo, opening book).
        case rewind([Int])
    }

    /// Reconciles the engine with `state` and, if it is the AI's turn, searches
    /// and plays.
    ///
    /// Cancellation works here and did not in the previous `Task.detached`
    /// version: a detached task is a cancellation root, so the `Task.isCancelled`
    /// the search polled was never the one `cancel()` had been called on. This
    /// `Task` is the one we hold, and `AZPlayer.think` polls it every 32
    /// simulations, so a 1600-simulation search now stops within ~20 ms.
    private func runEngine(sync: EngineSync) {
        guard let player else { return }
        let previous = engineTask
        previous?.cancel()

        let sims = level.rawValue
        let willThink = !gameOver && state.toPlay != humanSide
        thinking = willThink
        progressText = ""
        engineGeneration &+= 1
        let generation = engineGeneration

        engineTask = Task { [weak self] in
            _ = await previous?.value          // let the cancelled search unwind
            // The sync runs even when this task is already cancelled. Skipping
            // it would leave the actor's tree on a position the board no longer
            // shows, and `.rewind` is the only thing that ever repairs that --
            // a dropped one is never made up. It is cheap and idempotent, so
            // there is nothing to gain by bailing out first.
            switch sync {
            case .advance(let a): await player.play(a)
            case .rewind(let line): await player.rewind(to: line)
            }
            guard let self, willThink, !Task.isCancelled else { return }
            await self.search(with: player, sims: sims, generation: generation)
        }
    }

    private func search(with player: AZPlayer, sims: Int, generation: Int) async {
        let t0 = Date()
        /// True while this task is still the one the view model is driving.
        func current() -> Bool { generation == engineGeneration }

        status = sims > 0 ? "AI 思考中（\(sims) 次模拟）…" : "AI 思考中…"

        let result: SearchResult?
        do {
            result = try await player.think(
                simulations: sims,
                progress: { [weak self] done, total in
                    Task { @MainActor in
                        guard let self, generation == self.engineGeneration else { return }
                        self.progressText = "\(done)/\(total)"
                    }
                })
        } catch {
            result = nil
        }

        guard current() else { return }
        thinking = false
        progressText = ""
        guard let r = result, r.action >= 0 else {
            status = "引擎异常，请开新对局。"
            return
        }

        // Advance the tree before the move reaches the screen. Once `apply`
        // runs the board shows a stone the engine would not have; the ordering
        // here is what keeps the two from ever disagreeing, rather than relying
        // on the caller-side task chain to paper over the window.
        await player.play(r.action)
        guard current() else { return }

        lastMoveMs = Int(Date().timeIntervalSince(t0) * 1000)
        heat = r.visits.isEmpty ? nil : r.visits
        // `value` is the mover's view, and the mover is the AI until `apply`
        let aiIsBlack = state.toPlay == 1
        valueBlack = aiIsBlack ? r.value : -r.value
        apply(r.action)
        if !gameOver {
            let secs = String(format: "%.1f", Double(lastMoveMs ?? 0) / 1000)
            status = "AI 落子 \(Self.coordName(r.action))（\(secs)s）。轮到你。"
        }
    }

    /// Screenshot/UI-test hook: play a short scripted game.
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

    /// Board-coordinate label. Static and UI-only, so it reads the UI constant;
    /// `boot()` has already refused to run on a model whose board disagrees.
    nonisolated static func coordName(_ a: Int) -> String {
        let cols = Array("ABCDEFGHJKLMNOP")
        let r = a / Rules.board, c = a % Rules.board
        return "\(cols[c])\(Rules.board - r)"
    }
}

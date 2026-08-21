import SwiftUI

@main
struct Gomoku15App: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @StateObject private var vm = GameViewModel()

    var body: some View {
        GeometryReader { geo in
            let wide = geo.size.width > geo.size.height * 1.05
            Group {
                if wide {
                    HStack(alignment: .top, spacing: 16) {
                        BoardView(vm: vm)
                        panel.frame(minWidth: 280, maxWidth: 360)
                    }
                } else {
                    VStack(spacing: 12) {
                        BoardView(vm: vm)
                        panel
                    }
                }
            }
            .padding(14)
        }
        .background(Color(.systemGroupedBackground))
    }

    private var panel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("AlphaZero 五子棋").font(.title3.bold())
                Text("模型 iter040 · 10.5M 参数 · CoreML fp16 · 推理与搜索全在本机")
                    .font(.caption).foregroundStyle(.secondary)
                Label(vm.engineBadge, systemImage: vm.engineOK ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(vm.engineOK ? .green : .red)

                HStack {
                    Text("执子").font(.subheadline).foregroundStyle(.secondary)
                    Picker("执子", selection: Binding(
                        get: { vm.humanSide },
                        set: { vm.setSide($0) })) {
                        Text("我执黑（先手）").tag(Int8(1))
                        Text("我执白").tag(Int8(-1))
                    }
                    .pickerStyle(.segmented)
                    .disabled(vm.thinking || (!vm.moves.isEmpty && !vm.gameOver))
                }
                HStack {
                    Text("强度").font(.subheadline).foregroundStyle(.secondary)
                    Picker("强度", selection: $vm.level) {
                        ForEach(GameViewModel.Level.allCases) { l in
                            Text(l.label).tag(l)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                HStack(spacing: 10) {
                    Button(action: { vm.newGame() }) {
                        Label("新对局", systemImage: "arrow.counterclockwise")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.thinking)
                    Button(action: { vm.undo() }) {
                        Label("悔棋", systemImage: "arrow.uturn.backward")
                    }
                    .buttonStyle(.bordered)
                    .disabled(vm.thinking || vm.moves.isEmpty)
                    Button(action: { vm.showHeat.toggle() }) {
                        Label("AI 视角", systemImage: "eye")
                    }
                    .buttonStyle(.bordered)
                    .disabled(vm.heat == nil)
                }

                HStack(spacing: 8) {
                    if vm.thinking { ProgressView().controlSize(.small) }
                    Text(vm.status + (vm.progressText.isEmpty ? "" : "（\(vm.progressText)）"))
                        .font(.subheadline)
                        .foregroundStyle(vm.statusIsGood == true ? .green
                                         : vm.statusIsGood == false ? .red : .primary)
                }

                if let v = vm.valueBlack {
                    let p = Double(v + 1) / 2
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("黑方胜率（模型价值头）").font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Text("\(Int((p * 100).rounded()))%").font(.caption.monospacedDigit())
                            if let ms = vm.lastMoveMs {
                                Text("· \(ms) ms/手").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        GeometryReader { g in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Color(.systemGray5))
                                Capsule().fill(.black)
                                    .frame(width: g.size.width * p)
                            }
                        }
                        .frame(height: 8)
                    }
                }

                if !vm.moves.isEmpty {
                    Text(movesText)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                        .background(Color(.secondarySystemGroupedBackground),
                                    in: RoundedRectangle(cornerRadius: 8))
                }
                Text("「AI 视角」显示上一手搜索的访问热度。模型来自 node09 8×H20 上的 AlphaZero 自我对弈训练（40 轮，循环赛 Elo +1,639）。")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }

    private var movesText: String {
        vm.moves.enumerated()
            .map { i, a in "\(i + 1).\(i % 2 == 0 ? "●" : "○")\(GameViewModel.coordName(a))" }
            .joined(separator: "  ")
    }
}

import SwiftUI

/// Opening-book browser: mini board previews + stats; tap loads the line
/// into the game and returns.
struct BookView: View {
    let book: OpeningBook
    @ObservedObject var vm: GameViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var fullLine = false

    var body: some View {
        List {
            Section {
                Picker("摆到", selection: $fullLine) {
                    Text("前 \(book.openings.first?.ply_book ?? 8) 手").tag(false)
                    Text("全部主变").tag(true)
                }
                .pickerStyle(.segmented)
            }
            Section {
                ForEach(book.openings) { op in
                    Button {
                        vm.startFromOpening(op.line,
                                            plies: fullLine ? op.line.count : op.ply_book)
                        dismiss()
                    } label: {
                        HStack(spacing: 12) {
                            BoardCanvas(board: Self.stones(op.line),
                                        lastMove: op.line.last)
                                .frame(width: 84, height: 84)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(op.id.replacingOccurrences(of: "op", with: "#")) \(op.name)")
                                    .font(.subheadline.bold())
                                Text("黑胜率 \(String(format: "%.1f", op.winrate_black * 100))% · \(op.n) 局")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                if let v = op.v_black?.last {
                                    Text("模型评估 黑 \(Int(((v + 1) / 2 * 100).rounded()))%")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Text(lineText(op))
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.tertiary)
                                    .lineLimit(2)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            } footer: {
                Text("模型自我对弈中最常走且黑方战绩最好的开局（8 重对称归一统计）。这是 iter040 的\"开局观\"而非客观棋理：黑方总胜率 \(Int(((book.source.black_overall_winrate ?? 0) * 100).rounded()))%，高黑胜率首先反映先手优势。点击任意一条摆上棋盘接着下。")
            }
        }
        .navigationTitle("开局库")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func lineText(_ op: OpeningBook.Opening) -> String {
        op.line.enumerated()
            .map { i, a in (i % 2 == 0 ? "●" : "○") + GameViewModel.coordName(a) }
            .joined(separator: " ")
    }

    static func stones(_ line: [Int]) -> [Int8] {
        var b = [Int8](repeating: 0, count: Rules.cells)
        var p: Int8 = 1
        for a in line { b[a] = p; p = -p }
        return b
    }
}

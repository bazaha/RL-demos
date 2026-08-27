import SwiftUI

/// Compact candidate list: rank / coord / winrate (mover POV) / visits /
/// prior, plus the PV as coordinate text. Row tap toggles the board preview.
struct CandidateTable: View {
    let analysis: Analysis
    let selected: Int?
    let onSelect: (Int) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("候选着法（\(analysis.toPlay == 1 ? "黑" : "白")方视角胜率）· 点行切换棋盘上的 2 步预览")
                .font(.caption2).foregroundStyle(.secondary)
            ForEach(Array(analysis.candidates.enumerated()), id: \.element.id) { i, c in
                Button { onSelect(c.move) } label: {
                    HStack(spacing: 8) {
                        Text("#\(i + 1)")
                            .font(.caption2.monospaced()).foregroundStyle(.tertiary)
                        Text(GameViewModel.coordName(c.move))
                            .font(.caption.monospaced().bold())
                            .frame(width: 34, alignment: .leading)
                        Text("\(Int((Double(c.winrateMover) * 100).rounded()))%")
                            .font(.caption.monospacedDigit())
                            .frame(width: 40, alignment: .trailing)
                        Text("\(c.visitsN) 访问")
                            .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        Text(String(format: "先验 %.0f%%", c.prior * 100))
                            .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        Spacer(minLength: 4)
                        Text(c.pv.map(GameViewModel.coordName).joined(separator: "→"))
                            .font(.caption2.monospaced()).foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }
                    .padding(.vertical, 3).padding(.horizontal, 6)
                    .background(selected == c.move ? Color.teal.opacity(0.14) : .clear,
                                in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
            }
        }
    }
}

/// Hand-drawn black-winrate trend over the game (no chart library, matching
/// the BoardCanvas style). x = ply, y = 0-100% black POV.
struct WinrateChart: View {
    let history: [Int: Float]   // ply -> vBlack in [-1, 1]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("黑方胜率走势").font(.caption2).foregroundStyle(.secondary)
            Canvas { ctx, size in
                let pts = history.sorted { $0.key < $1.key }
                guard let maxPly = pts.last?.key else { return }
                let w = size.width, h = size.height
                for frac in [0.0, 0.5, 1.0] {
                    var g = Path()
                    g.move(to: CGPoint(x: 0, y: h * frac))
                    g.addLine(to: CGPoint(x: w, y: h * frac))
                    ctx.stroke(g, with: .color(.gray.opacity(frac == 0.5 ? 0.45 : 0.25)),
                               style: StrokeStyle(lineWidth: 1,
                                                  dash: frac == 0.5 ? [4, 3] : []))
                }
                func at(_ ply: Int, _ v: Float) -> CGPoint {
                    let x = maxPly == 0 ? 0 : w * CGFloat(ply) / CGFloat(maxPly)
                    let p = CGFloat(v + 1) / 2          // black win prob
                    return CGPoint(x: x, y: h * (1 - p))
                }
                if pts.count > 1 {
                    var line = Path()
                    line.move(to: at(pts[0].key, pts[0].value))
                    for (k, v) in pts.dropFirst() { line.addLine(to: at(k, v)) }
                    ctx.stroke(line, with: .color(.teal),
                               style: StrokeStyle(lineWidth: 1.6, lineJoin: .round))
                }
                for (k, v) in pts {
                    let p = at(k, v)
                    ctx.fill(Path(ellipseIn: CGRect(x: p.x - 2.2, y: p.y - 2.2,
                                                    width: 4.4, height: 4.4)),
                             with: .color(.teal))
                }
            }
            .frame(height: 64)
            HStack {
                Text("0 手").font(.caption2).foregroundStyle(.tertiary)
                Spacer()
                if let last = history.max(by: { $0.key < $1.key }) {
                    Text("第 \(last.key) 手 · 黑 \(Int((Double(last.value + 1) / 2 * 100).rounded()))%")
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
        }
    }
}

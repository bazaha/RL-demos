import SwiftUI

struct BoardView: View {
    @ObservedObject var vm: GameViewModel

    var body: some View {
        GeometryReader { geo in
            let size = min(geo.size.width, geo.size.height)
            Canvas { ctx, _ in
                draw(ctx: ctx, size: size)
            }
            .frame(width: size, height: size)
            .contentShape(Rectangle())
            .onTapGesture { pt in
                let B = Rules.board
                let pad = size * 0.045
                let cell = (size - 2 * pad) / CGFloat(B - 1)
                let c = Int(((pt.x - pad) / cell).rounded())
                let r = Int(((pt.y - pad) / cell).rounded())
                guard r >= 0, r < B, c >= 0, c < B else { return }
                let dx = pt.x - (pad + CGFloat(c) * cell)
                let dy = pt.y - (pad + CGFloat(r) * cell)
                guard dx * dx + dy * dy <= cell * cell * 0.45 else { return }
                vm.tap(r * B + c)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private func draw(ctx: GraphicsContext, size: CGFloat) {
        let B = Rules.board
        let pad = size * 0.045
        let cell = (size - 2 * pad) / CGFloat(B - 1)
        let boardColor = Color(red: 0.91, green: 0.85, blue: 0.71)
        let lineColor = Color(red: 0.63, green: 0.55, blue: 0.38)
        func xy(_ a: Int) -> CGPoint {
            CGPoint(x: pad + CGFloat(a % B) * cell, y: pad + CGFloat(a / B) * cell)
        }

        ctx.fill(Path(roundedRect: CGRect(x: 0, y: 0, width: size, height: size),
                      cornerRadius: 12), with: .color(boardColor))
        var grid = Path()
        for i in 0..<B {
            let o = pad + CGFloat(i) * cell
            grid.move(to: CGPoint(x: pad, y: o)); grid.addLine(to: CGPoint(x: size - pad, y: o))
            grid.move(to: CGPoint(x: o, y: pad)); grid.addLine(to: CGPoint(x: o, y: size - pad))
        }
        ctx.stroke(grid, with: .color(lineColor), lineWidth: 0.7)
        for (r, c) in [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)] {
            let p = xy(r * B + c)
            ctx.fill(Path(ellipseIn: CGRect(x: p.x - 2.5, y: p.y - 2.5, width: 5, height: 5)),
                     with: .color(lineColor))
        }

        // AI-view heatmap under the stones
        if vm.showHeat, let heat = vm.heat {
            let mx = heat.max() ?? 0
            if mx > 0 {
                for i in 0..<Rules.cells where heat[i] >= max(2, mx * 0.08) && vm.board[i] == 0 {
                    let p = xy(i)
                    let alpha = 0.08 + 0.55 * Double(heat[i] / mx)
                    ctx.fill(Path(roundedRect: CGRect(x: p.x - cell * 0.38, y: p.y - cell * 0.38,
                                                      width: cell * 0.76, height: cell * 0.76),
                                  cornerRadius: 4),
                             with: .color(.blue.opacity(alpha)))
                }
            }
        }

        for i in 0..<Rules.cells where vm.board[i] != 0 {
            let p = xy(i)
            let rad = cell * 0.44
            let rect = CGRect(x: p.x - rad, y: p.y - rad, width: 2 * rad, height: 2 * rad)
            let isBlack = vm.board[i] == 1
            ctx.fill(Path(ellipseIn: rect),
                     with: .radialGradient(
                        Gradient(colors: isBlack
                                 ? [Color(white: 0.28), .black]
                                 : [.white, Color(white: 0.88)]),
                        center: CGPoint(x: p.x - rad * 0.3, y: p.y - rad * 0.35),
                        startRadius: rad * 0.1, endRadius: rad * 1.2))
            ctx.stroke(Path(ellipseIn: rect),
                       with: .color(.black.opacity(isBlack ? 0.5 : 0.3)), lineWidth: 0.6)
        }

        if let last = vm.moves.last {
            let p = xy(last)
            ctx.stroke(Path(ellipseIn: CGRect(x: p.x - cell * 0.18, y: p.y - cell * 0.18,
                                              width: cell * 0.36, height: cell * 0.36)),
                       with: .color(.orange), lineWidth: 2)
        }
        if let win = vm.winCells, let f = win.first, let l = win.last {
            var line = Path()
            line.move(to: xy(f)); line.addLine(to: xy(l))
            ctx.stroke(line, with: .color(.orange),
                       style: StrokeStyle(lineWidth: 3.5, lineCap: .round))
        }
    }
}

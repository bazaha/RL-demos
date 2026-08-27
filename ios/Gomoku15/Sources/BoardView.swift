import SwiftUI

/// Pure display board: no view-model dependency, reused by the game screen,
/// the opening-book previews and the analysis overlay.
struct BoardCanvas: View {
    var board: [Int8]
    var lastMove: Int? = nil
    var winCells: [Int]? = nil
    var heat: [Float]? = nil
    var showHeat = false
    var candidates: [Candidate]? = nil   // analysis marks (empty cells only)
    var pvLine: [Int]? = nil             // ghost-stone preview of one PV
    var pvToPlay: Int8 = 1               // mover at the analyzed position

    var body: some View {
        Canvas { ctx, size in
            Self.draw(ctx: ctx, size: min(size.width, size.height),
                      board: board, lastMove: lastMove, winCells: winCells,
                      heat: heat, showHeat: showHeat,
                      candidates: candidates, pvLine: pvLine,
                      pvToPlay: pvToPlay)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    static func draw(ctx: GraphicsContext, size: CGFloat, board: [Int8],
                     lastMove: Int?, winCells: [Int]?, heat: [Float]?,
                     showHeat: Bool, candidates: [Candidate]? = nil,
                     pvLine: [Int]? = nil, pvToPlay: Int8 = 1) {
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

        // heat (last AI search) — suppressed while candidate marks are shown,
        // both live on empty cells and would overpaint each other
        if showHeat, let heat, candidates == nil {
            let mx = heat.max() ?? 0
            if mx > 0 {
                for i in 0..<Rules.cells where heat[i] >= max(2, mx * 0.08) && board[i] == 0 {
                    let p = xy(i)
                    let alpha = 0.08 + 0.55 * Double(heat[i] / mx)
                    ctx.fill(Path(roundedRect: CGRect(x: p.x - cell * 0.38, y: p.y - cell * 0.38,
                                                      width: cell * 0.76, height: cell * 0.76),
                                  cornerRadius: 4),
                             with: .color(.blue.opacity(alpha)))
                }
            }
        }

        for i in 0..<Rules.cells where board[i] != 0 {
            stoneBody(ctx, at: xy(i), rad: cell * 0.44, isBlack: board[i] == 1)
        }

        // analysis candidate marks: winrate% (mover POV) + visits
        if let cands = candidates {
            for (idx, c) in cands.enumerated() where board[c.move] == 0 {
                let p = xy(c.move)
                let rad = cell * 0.44
                let rect = CGRect(x: p.x - rad, y: p.y - rad,
                                  width: 2 * rad, height: 2 * rad)
                let isBest = idx == 0
                let tint = isBest ? Color(red: 0.05, green: 0.55, blue: 0.47)
                                  : Color(red: 0.36, green: 0.47, blue: 0.66)
                ctx.fill(Path(ellipseIn: rect),
                         with: .color(tint.opacity(isBest ? 0.92 : 0.60)))
                ctx.stroke(Path(ellipseIn: rect), with: .color(tint),
                           lineWidth: isBest ? 1.6 : 0.8)
                let wr = Int((Double(c.winrateMover) * 100).rounded())
                if cell >= 22 {
                    ctx.draw(Text("\(wr)")
                        .font(.system(size: cell * 0.34, weight: .bold))
                        .foregroundColor(.white),
                             at: CGPoint(x: p.x, y: p.y - cell * 0.11))
                    ctx.draw(Text("\(c.visitsN)")
                        .font(.system(size: cell * 0.22))
                        .foregroundColor(.white.opacity(0.85)),
                             at: CGPoint(x: p.x, y: p.y + cell * 0.20))
                } else {
                    ctx.draw(Text("\(wr)")
                        .font(.system(size: cell * 0.36, weight: .bold))
                        .foregroundColor(.white), at: p)
                }
            }
        }

        // PV ghost stones: the selected candidate plus the next plies
        if let pv = pvLine {
            for (k, a) in pv.enumerated() where board[a] == 0 {
                let isBlack = (k % 2 == 0 ? pvToPlay : -pvToPlay) == 1
                let p = xy(a)
                var ghost = ctx
                ghost.opacity = 0.62
                stoneBody(ghost, at: p, rad: cell * 0.44, isBlack: isBlack)
                ghost.draw(Text("\(k + 1)")
                    .font(.system(size: cell * 0.40, weight: .semibold))
                    .foregroundColor(isBlack ? .white : .black), at: p)
            }
        }

        if let last = lastMove {
            let p = xy(last)
            ctx.stroke(Path(ellipseIn: CGRect(x: p.x - cell * 0.18, y: p.y - cell * 0.18,
                                              width: cell * 0.36, height: cell * 0.36)),
                       with: .color(.orange), lineWidth: 2)
        }
        if let win = winCells, let f = win.first, let l = win.last {
            var line = Path()
            line.move(to: xy(f)); line.addLine(to: xy(l))
            ctx.stroke(line, with: .color(.orange),
                       style: StrokeStyle(lineWidth: 3.5, lineCap: .round))
        }
    }

    private static func stoneBody(_ ctx: GraphicsContext, at p: CGPoint,
                                  rad: CGFloat, isBlack: Bool) {
        let rect = CGRect(x: p.x - rad, y: p.y - rad, width: 2 * rad, height: 2 * rad)
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
}

/// Interactive board for the game screen: BoardCanvas + the tap gesture.
struct BoardView: View {
    @ObservedObject var vm: GameViewModel

    var body: some View {
        GeometryReader { geo in
            let size = min(geo.size.width, geo.size.height)
            BoardCanvas(board: vm.board, lastMove: vm.moves.last,
                        winCells: vm.winCells, heat: vm.heat, showHeat: vm.showHeat,
                        candidates: vm.analysisOn ? vm.analysis?.candidates : nil,
                        pvLine: vm.analysisOn ? vm.previewPV : nil,
                        pvToPlay: vm.analysis?.toPlay ?? 1)
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
}

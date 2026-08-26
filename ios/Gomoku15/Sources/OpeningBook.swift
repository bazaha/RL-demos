import Foundation

/// The opening book mined from mass self-play (results/book pipeline).
/// Bundled as gomoku_book.json; schema shared with the web play page.
struct OpeningBook: Decodable {
    struct Source: Decodable {
        let games: Int
        let temp_moves: Int?
        let black_overall_winrate: Double?
    }
    struct Opening: Decodable, Identifiable {
        let id: String
        let name: String
        let line: [Int]
        let ply_book: Int
        let n: Int
        let black_wins: Int
        let white_wins: Int
        let draws: Int
        let winrate_black: Double
        let wilson_lb: Double
        let avg_len: Double
        let line_support: [Int]
        let v_black: [Double]?
    }

    let version: Int
    let board: Int
    let ckpt: String
    let source: Source
    let openings: [Opening]

    static func load() -> OpeningBook? {
        guard let url = Bundle.main.url(forResource: "gomoku_book", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let book = try? JSONDecoder().decode(OpeningBook.self, from: data),
              book.board == Rules.board, !book.openings.isEmpty else { return nil }
        return book
    }
}

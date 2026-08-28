# GomokuEngine — 原生 iOS/macOS 推理引擎骨架

`iter040.pt`（15×15 AlphaZero，10.53M 参数）的原生对战引擎：Core ML 在 ANE 上做推理，
Swift 复刻训练器的 MCTS。**只有引擎，没有 UI** —— 规则、推理、搜索、自检，接进任何 SwiftUI/UIKit 界面即可。

浏览器版（`report/gomoku_play.html`）单次前向 78 ms，同一台机器上这条路径 0.66 ms。
差的两个数量级全在推理实现上，不在模型上。

## 快速开始

```bash
uv sync --group coreml
AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 uv run python scripts/export_gomoku_coreml.py
```

导出脚本会做三件事并写进 `results/coreml_export/coreml_report.json`：对拍 `testvec.json` 的
5 个参考局面、检查每个算子是否都落在 ANE 上、量各 compute unit 的延迟。**三项都过才算可用**。

然后跑引擎测试（`-c release` 不是可选项，见下）：

```bash
GOMOKU_MODEL=$PWD/results/coreml_export/GomokuAZ_b1.mlpackage \
GOMOKU_TESTVEC=$PWD/results/coreml_export/testvec.json \
swift test -c release --package-path ios/GomokuEngine
```

不设 `GOMOKU_MODEL` 时，依赖模型的测试自动 skip，规则/搜索的测试照常跑。

## 文件

| 文件 | 内容 |
| --- | --- |
| `GomokuState.swift` | 棋盘、落子、胜负判定、4 平面编码。struct，所以训练器里显式的 `clone()` 在这里就是赋值 |
| `AZNet.swift` | Core ML 封装：加载、fp16 输入复用、合法掩码 softmax |
| `MCTS.swift` | `Node` / `Tree`：PUCT 选择、展开、逐层翻号回传、子树复用 |
| `AZPlayer.swift` | actor，持有棋局与搜索树，`think()` 在后台跑，支持温度采样与取消 |
| `EngineSelfTest.swift` | 用 `testvec.json` 做端到端自检，**建议接到 App 启动流程里**，不要只当单元测试 |

## 六条不能走样的语义

移植 AlphaZero 最容易错的地方，网页版当初逐条踩过一遍（见 `CLAUDE.md`）：

1. **`terminal_value` 是"当前该走棋的一方"的视角**。刚赢棋的一方在终局节点看到的是 −1，因为轮到的是输家
2. **`backup` 每上一层翻一次号，先翻再累加**。路径最底端那条边属于叶子的父节点，也就是叶子走子方的对手
3. **根节点的 `W/N` 已经是根方视角**，读出来给 UI 显示时**不要再取负**
4. **PUCT 里 `q` 在 `N==0` 时取 0**，不是取父节点的值，也不是 −1
5. **展开时先掩合法再归一化**。网络输出的是原始 logits，掩码必须在 softmax 之前
6. **温度采样先除以最大访问数再取 1/τ 次幂**。反过来做，小温度会溢出

`GomokuEngineTests` 里 1、2、5 各有一条断言，改这几处会立刻红。

## 两个实测出来的坑

**必须显式要 `.cpuAndNeuralEngine`。** 用默认的 `.all`，Core ML 的调度器会挑 GPU：
M2 Ultra 上 batch=1 是 0.665 ms（ANE）对 2.86 ms（GPU / `.all`），4.3 倍。
`AZNet.load` 默认值已经设好，但如果你改用 Xcode 自动生成的模型类，记得传 `MLModelConfiguration`。

**Swift 一定要 release 编译。** 同一份代码同一台机器，400 次模拟：debug 469 ms、release 264 ms。
release 下每次模拟 0.661 ms，和裸推理的 0.665 ms 基本相等 —— **搜索树的开销已经小到测不出来**，
瓶颈完全是网络。这也意味着没必要为了性能去做 virtual-loss 批量搜索。

## 实测数字（M2 Ultra 60 核 GPU / 32 核 ANE）

| | batch=1 | batch=8 |
| --- | --- | --- |
| Core ML ANE | **0.665 ms** | 3.43 ms（0.43 ms/局面）|
| Core ML GPU | 2.86 ms | 10.9 ms |
| Core ML CPU | 2.67 ms | 9.57 ms |
| 端到端 400 次模拟一手（release） | **264 ms** | — |

模型加载：ANE 0.87 s，GPU 0.21 s。**ANE 的加载明显更慢，放到 App 启动时预热，别等第一手棋。**

设备换算（按 ANE 规格与内存带宽估算，未在真机验证）：iPad Pro M5 每手 0.2–0.4 s，
iPhone 16 Pro Max 每手 0.4–0.8 s。`testSearchThroughput` 就是拿来在真机上把这两个估算换成实测的。

## 没有包含的

- **UI**：棋盘绘制、落子交互、悔棋、AI 视角热度图
- **批量叶子评估**：`AZNet.evaluate` 支持批量（导出时加 `CML_BATCHES=1,8`），但 `AZPlayer` 走串行。
  ANE 上 batch=1 只比 batch=8 差 1.6 倍，除非要冲很高的模拟数，否则不值得引入 virtual loss 的复杂度
- **开局库 / 让子**
- **悔棋的树复用**：`rewind(to:)` 直接重建树。搜索树没法往回走，撤销就是重放

## 接进 App

把 `.mlpackage` 拖进 Xcode target（Xcode 会自动编译成 `.mlmodelc` 并放进 bundle），
`testvec.json` 作为资源一起打包，然后：

```swift
let url = Bundle.main.url(forResource: "GomokuAZ_b1", withExtension: "mlmodelc")!
let net = try await AZNet.load(url: url)                 // 默认走 ANE
let results = try EngineSelfTest.run(net: net, testVectorURL: vecURL)
assert(results.allSatisfy(\.passed))                     // 启动自检

let player = AZPlayer(net: net)
await player.play(humanMove)
let move = try await player.think(simulations: 400)      // 不阻塞 UI
```

`AZPlayer` 是 actor：模型、棋局、搜索树三者始终一致，UI 永远不会被搜索卡住。
搜索循环本身在一次 actor hop 里同步跑完 —— 每次模拟一个 await 的话，光调度开销就超过搜索本身。

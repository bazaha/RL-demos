# STATUS.md — 任务与训练状态快照

> 快照时间：**2026-08-18**。本文件是跨机器交接用的状态账本：**做到哪了、产物在哪、接下来做什么**。
> "怎么做"（两机工作流、镜像、坑）的权威在 [CLAUDE.md](CLAUDE.md)，本文件不重复，只在需要处引用。
> 在新机器上恢复工作：直接读本文件 §4；让 Claude Code 接手时，它会自动加载 CLAUDE.md，再把本文件读一遍即可获得全部上下文。

## 1. 一页总览

| 工作流 | 状态 | 交付物 | 说明 |
| --- | --- | --- | --- |
| 通路验证（ResNet-18/CIFAR-10 + DQN/CartPole） | ✅ 完成 2026-07-24 | `report/index.html` | 验证环境通、能训练、可展示 |
| 9×9 AlphaZero 双臂 A/B（规则冷启动） | ✅ 完成 2026-07-25 | `report/gomoku.html` | 结论：该规模测不出冷启动优势（50.9%±3.9pp） |
| Phase-0 标定（和棋诊断/吞吐/15×15 成本） | ✅ 完成 2026-07-31 | `results/calib_*.jsonl` | 三结论见 CLAUDE.md「Phase-0 标定」 |
| Phase-2 评估体系（温度采样/锚点池/分级战术题/断点续跑） | ✅ 完成 2026-07-31~08-17 | trainer 内 | 全部向后兼容,老 JSON 照常渲染 |
| **Phase-1 主线：15×15 单臂 40 轮** | ✅ 完成 2026-08-17 | `report/gomoku15.html` | Elo +1,639;含一次已验证的训练病理与干预,见 §2 |
| 人机对战页（浏览器内推理+MCTS） | ✅ 完成 2026-08-18 | `report/gomoku_play.html`（28 MB） | 引擎与训练端对拍 ≤2e-6 |
| 本地推理服务（Mac MPS / node09 CUDA docker） | ✅ 完成 2026-08-18 | `scripts/serve_gomoku.py` + `run_gomoku_serve.sh` | 页面自动探测,解锁 1600 sims;400 sims ≈ 0.8 s |
| 离线硬探针（检查点战术补测） | ✅ 完成 2026-08-18 | `results/gomoku_hard_probes.json` + `report/gomoku_probes.html` | 浅战术 iter5 饱和是真实能力;value 校准与风格拐点见 §3 |
| VCF 求解器基线（替代 pure-MCTS） | ✅ 完成 2026-08-19 | `results/gomoku_vcf_baseline.json` | 量程 30 轮、中段有结构,见 §3 |
| iOS 人机对战 App（CoreML/ANE） | ✅ 完成 2026-08-21（分支 `ios-app`） | `ios/`（Xcode 工程 + mlpackage） | 引擎对拍全绿;真机 ANE 数字待装机读取 |
| Phase-3 吞吐 / Phase-4 A/B v2 | ⏳ 未开始 | — | backlog 见 §3 |

当前 node09 上跑着一个容器：`az_serve`（推理服务,GPU 0,只绑回环,`docker rm -f az_serve` 可停）；GPU 5 长期被其他用户占用（49 GB,避开）。本地无定时任务/监控残留。

## 2. 训练状态（截至快照）

### 2.1 Phase-1（15×15,tag=`p15`）——最新完成的主线

- 配置：15×15 / 192ch / 12blk（10.53M 参数）/ 800 sims / 每轮 1,344 局 / 40 轮 / seed 42 / β₀=0
- 结果：终局温度采样循环赛 **Elo +1,639**（±55）;pure-MCTS-400/8000 与 rule-greedy 全部打满;总墙钟 10.7 h（含两次拓扑切换空档）
- **过程时间线**（读 `report/gomoku15.html` 或 `results/gomoku_p15.json` 可复核）：
  - iter 1-4：正常起步（28 workers/7 卡,~22 min/轮,比标定外推慢 2 倍——每卡 4 进程 contention）
  - iter 5-25：**短对局塌缩**（手数 58→11-14,黑胜 96%,零和棋）。锚点梯子是唯一报警的仪表:5 轮间隔得分 92%→67%→58%→50%（停滞）
  - iter 20：按用户要求 28→12 workers（3 卡）,借新加的 `AZ_RESUME_ITER` 无损续跑
  - **iter 25：干预**——`AZ_TEMP_MOVES` 20→10（诊断:温度窗盖满全局,白方强制防守被采样噪声打断,黑方刷廉价速胜）。3 轮内手数→24、白胜×5、出现和棋;锚点恢复 75%→83%;循环赛证实泥潭期 15 轮仅 +157 Elo,干预后 5 轮 **+386**
  - iter 35-40：平台（锚点 vs iter035 = 33%±27pp）,40 轮收官是合理停点
- 已知非致命瑕疵：`resumed_at` 多次续跑只留最后一条（修法见 §3）

### 2.2 检查点与数据位置

| 内容 | 位置 | 大小 | 丢了能再生吗 |
| --- | --- | --- | --- |
| **p15 检查点 ×9**（iter000-040,每 5 轮） | node09 `~/h20_validation_20260724/results/gomoku_ckpt_p15/` | 362 MB | ❌ 唯一副本（仅 iter040.pt 另有本地副本,供推理服务用,git 忽略） |
| pure/rules 检查点（9×9 各 11 个） | node09 同目录 `gomoku_ckpt_{pure,rules}/` | 123 MB ×2 | ❌ 唯一副本 |
| 训练指标 JSON / GPU 日志 / 标定 jsonl | 本地 `results/` 与 node09 双份 | ~2.5 MB | 双份互备 |
| 网页权重导出（fp16 + manifest + 参考向量） | 本地 `results/web_export/` 与 node09 双份 | 21 MB | ✅ 可由 iter040.pt 重导 |
| 五份报告 | 本地 `report/` | 29 MB（play 占 28） | ✅ 全部可由 results/ + scripts/ 再生 |
| `gpu_log_gomoku_p15.csv` | 本地（**已合并** part1+part2+尾段,35,197 样本） | 1.0 MB | node09 上仍是三段,本地这份是权威 |

## 3. 任务账本

### 已完成（按时间）

- [x] 2026-07-24 通路验证 demo + 报告
- [x] 2026-07-25 9×9 双臂训练、臂间对打评估、`report/gomoku.html`（含测量分辨率修正:臂内梯子只排顺序等,见 CLAUDE.md）
- [x] 2026-07-31 Phase-0 三组标定;Phase-2 评估体系（AZPlayer 温度采样、锚点池、分级战术题+精确校验器、`run_gomoku15_in_container.sh`）;报告模板新卡与单臂容错;战术校验器"对手无成五点"加固（随机化对抗复核 1,794 声明 0 驳倒）
- [x] 2026-08-17 Phase-1 40 轮全程（含 2 次在线换配置的断点续跑、1 次温度干预）;`AZ_RESUME_ITER` 断点续跑;`report/gomoku15.html`;发射前多智能体审查修掉 5 处报告硬编码
- [x] 2026-08-21 iOS App（分支 `ios-app`,`ios/` 目录）:iter040 → CoreML fp16 mlprogram(`export_gomoku_coreml.py`,Mac 上对拍 5 向量全绿,最差 1.8e-3);Swift 移植 State/MCTS(与 trainer 逐语义一致)+ SwiftUI 双端自适应界面;XCTest 全绿(规则/CoreML parity/战术);iPhone 17 Pro Max 与 iPad Pro M5 模拟器截图验证。装真机:Xcode 开 `ios/Gomoku15.xcodeproj`,设置签名 Team 后 Run;App 内徽章显示引擎校验与每手 ms
- [x] 2026-08-19 VCF 求解器基线（`eval_gomoku_vcf_baseline.py`:成五>封五>VCF(5)>拆双威胁>规则贪心,全复用校验器;AZ 侧温度 0.3,每检查点 12 局）。得分曲线 0.00→0.50→0.42→0.42→0.92→0.92→1.00→0.92→1.00:**量程 30 轮**(rule-greedy 只有 15),且有结构——冲锋流期(iter10-15)反而输给纯战术机器(0.42),iter20 起进攻深度超出其 2 手防守视界。强网络的零星败局是长对局末段漏掉 5 深 VCF(iter035 败局:45 手,第 39 手失守),**400 sims 下的 VCF 盲区真实存在但罕见**。`resumed_at` 一行修复同批完成
- [x] 2026-08-18 离线硬探针（`eval_gomoku_hard_probes.py`,4 族 × 2 向,诱饵与正解分离、构造期校验器证明）。三个发现:①浅战术（≤3 手强制,含毒化冲四）iter5 起 raw 全对、零上钩——饱和是真实能力,此后的 Elo 增长不在浅战术里;②必胜局面的 value 置信是晚熟信号,+0.64(iter5)→+1.00(iter35),iter25_tr 曾出现"下对棋却判 -0.91";③HV2 风格拐点与 iter25 温度干预精确对齐:干预前全走直接双威胁 (5,10),干预后全走保先占毒点 (12,12),两者皆客观胜着。教训:判卷 good 集必须=全部客观胜着（_vcf_starts）,窄判卷曾把更聪明的下法误判成回退
- [x] 2026-08-18 本地推理服务（复用 trainer 的网络与 MCTS,页面探测/回退,MPS 与 CUDA 双部署,跨后端同权重同落子）
- [x] 2026-08-18 人机对战页全链路（导出→WebGL2 推理→JS MCTS→对拍验证→交互验证）,抓修 GPU GroupNorm 单遍方差、纹理单元 clobber、aiTurn 回合守卫、执白悔棋死局等 9 个 bug

### Backlog（剩余工作：是什么、解决什么问题）

**先说决策点（在 backlog 之外）**：要不要再训一个更强的模型（15×15 更多轮，或 19×19）。
要 → Phase-3 是第一步、Phase-4 可搭车；不要 → 下面 1、2 都无需启动，只剩第 3 项和"未列入"里的检查点备份两件小事。

**1. Phase-3 吞吐工程 —— 前三项已实现并标定（分支 `phase-3-throughput`，2026-08-21）**

trainer 新增环境变量开关（默认兼容旧行为）：`AZ_CAP_PROB/AZ_CAP_SIMS`（playout cap randomization，
只有全搜索的手被记录为训练目标并带根噪声）、`AZ_RESIGN/_V/_N/_MIN/_KEEP`（认输 + 16 手护栏 +
不认输对照组审计假认输率）、`AZ_DEAD_DRAW`（死和裁定，数学上无损，默认开）。
15×15/iter040/temp10 regime 的消融标定（`results/calib_phase3.jsonl`，单点噪声 ±20%）：

| 配置 | games/s | targets/s | 假认输 |
| --- | --- | --- | --- |
| cap 0.25 单独 | **2.69×** | 0.40× | — |
| cap 0.5 + 认输 + 死和 | 1.37× | 0.49× | 0/134 ✓ |
| cap 0.25 + 认输 + 死和 | 1.96× | 0.31× | 1/134 ✓ |
| 只认输（护栏 16 手） | ~1× | 0.81× | 0/134 ✓ |

要点：cap 是吞吐大头；**无护栏的认输会和网络的黑必胜偏见共振**（第 7-9 手大批早退,约 7% 错标——
`AZ_RESIGN_MIN=16` 把假认输压到 0-1/134）；本标定是"强网络短棋"regime,从零训练的早中期（手数 50-60）
cap 的目标饥荒会轻得多、认输收益更大。下次训练推荐 `CAP_PROB=0.25~0.33 + RESIGN=1(MIN 16) + DEAD_DRAW=1`,
并按记录目标数下调 `AZ_TRAIN_STEPS` 保持 4-8× 重用。**每卡单推理服务进程：已实现（`AZ_SERVED=1`,同步版）,实测判定不划算,默认关**。
served 模式下 worker 纯 CPU、权重只进 n_GPU 个 server、前向与本地逐位一致;但 12/24/48 worker、
小批窗、bf16 五个变体全部钉在 1.05-1.12 games/s（经典 12w = 1.34）——**同步逐 sim 往返（~3-5ms IPC）
成为每个 worker 的节拍器**,批量红利无从兑现,200+ 手长棋尾部在 batch≈1 下每 sim 都付全额往返。
要兑现该架构需 async 评估 + 虚拟损失（每树多个在途叶子,Tree/selfplay 重构）——记为后续真实前置。
**剩余**：async/虚拟损失重构（如需再上规模）+ 用一次从零短跑验证 cap/认输的训练质量不回退。

**2. Phase-4 冷启动 A/B v2 —— 解决"原始问题其实没答完"**

7 月的答案"测不出"是测量分辨率的失败，不是"没用"的证明。v2 修三个测量缺陷；
需要 6-9 个完整跑次，等 Phase-3 把单跑成本压下来再做：

- 先验没机会值钱：每轮 384 局数据太充裕 → 砍到 1/4，让早期数据稀缺
- 判据会饱和：固定基线 15 轮内全满 → 改比"到达固定强度用了几轮"，标尺用锚点 / VCF 基线这类不饱和的
- 种子方差从没测过：±3.9pp 只覆盖对局噪声，单种子运气可能比效应还大 → 每臂 ≥3 种子

**3. 人机对战页小增强 —— 纯产品体验，无研究价值**

AI 落子温度档（现在 argmax 确定性，同样下法必得同一局，会被单一克制线路刷穿；`AZPlayer` 的 `temp` 参数现成）、
开局库/让子、移动端触控（触屏没有 hover 幽灵子）。

### 未列入 backlog 但需要知道的

- **检查点备份风险**：iter040 之外的 8 个 p15 检查点 + 两组 9×9 检查点（约 570 MB）唯一副本在 node09（§2.2），节点数据丢失即永久丢失
- **VCT 求解器 / 外部引擎**：比 VCF 更长量程的绝对战术标尺的下一级（算活三、四三，工作量比 VCF 大一档）——只在"新训练跑需要不饱和标尺"时才值得做

## 4. 在另一台机器上继续

### 前置条件

1. **SSH 别名 `node09`** 可用（`node09.tx.bj.stonewise.cn`,用户 `fudong`,需配 key）——训练/导出的唯一硬依赖
2. 本地只要 `python3`（报告与页面生成**无任何第三方依赖**,不需要 torch）+ rsync
3. 验证报告/对战页需要 Chrome（headless 截图/自检,坑见 CLAUDE.md）

### 恢复步骤

```bash
git clone <repo> && cd rl-demos
ssh node09 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'  # 通路 + 看 GPU5 是否仍被占
# 按需取回数据（若 repo 未含 results/）：
rsync -az 'node09:~/h20_validation_20260724/results/*.json' 'node09:~/h20_validation_20260724/results/*.jsonl' 'node09:~/h20_validation_20260724/results/*.csv' results/
# 注意:gpu_log_gomoku_p15.csv 在 node09 是三段,合并逻辑见 CLAUDE.md「Phase-1 实跑记录」
```

### 命令速查（详情与坑全在 CLAUDE.md）

| 目的 | 命令 |
| --- | --- |
| 同步脚本上去 | `rsync -az --exclude __pycache__ scripts node09:~/h20_validation_20260724/` |
| 15×15 训练（新跑/续跑） | CLAUDE.md「Two-machine workflow」+ `run_gomoku15_in_container.sh`;续跑加 `-e AZ_RESUME_ITER=N` |
| 臂间对打（9×9 检查点） | `run_gomoku_cross_arm.sh` |
| 生成 9×9 双臂报告 | `python3 scripts/gen_gomoku_report.py` |
| 生成 15×15 单臂报告 | `GOMOKU_ARMS=p15 GOMOKU_OUT=report/gomoku15.html python3 scripts/gen_gomoku_report.py` |
| 人机对战页（权重已导出时） | `python3 scripts/gen_gomoku_play.py` |
| VCF 基线评测 | 容器内 `AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 python scripts/eval_gomoku_vcf_baseline.py` |
| 硬探针评测 + 可视化 | 容器内 `AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 python scripts/eval_gomoku_hard_probes.py`;本地 `python3 scripts/gen_probes_report.py` → `report/gomoku_probes.html` |
| 推理服务（Mac,需一次 `uv venv` 装 torch,见 CLAUDE.md） | `.venv-serve/bin/python scripts/serve_gomoku.py` |
| 推理服务（node09 + 隧道） | node09 上 `bash scripts/run_gomoku_serve.sh`;Mac 上 `ssh -N -L 8787:127.0.0.1:8787 node09` |
| 人机对战页（从检查点重导权重） | 容器内 `AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 CAL_CKPT=results/gomoku_ckpt_p15/iter040.pt python scripts/export_gomoku_web.py`,rsync 回 `results/web_export/` 再本地组装 |

## 5. git 托管建议

- **全部提交**（2026-08-20 起的策略,repo ≈ 100 MB）：`scripts/`、两个 md、`results/` 的 JSON/JSONL/CSV、五份报告（含 28 MB 的对战页）、`web_export/`（含 21 MB 权重）、**`iter040.pt`**（42 MB,最终模型——本地推理服务与页面重导出都靠它,repo 因此脱离 node09 也完整可用）
- **仍不进 git**：`data/`（CIFAR）;iter040 之外的 8 个检查点（320 MB,只在 node09——如担心节点数据丢失,先 `rsync` 一份 `gomoku_ckpt_p15/` 到可靠存储）;`__pycache__`/`.venv-serve` 等 scratch。实际规则见仓库根的 `.gitignore`

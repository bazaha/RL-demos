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
| 离线硬探针 / VCF 基线 / Phase-3 吞吐 / Phase-4 A/B v2 | ⏳ 未开始 | — | backlog 见 §3 |

当前 node09 上**没有**我们在跑的容器；GPU 5 长期被其他用户占用（49 GB,避开）。本地无定时任务/监控残留。

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
| **p15 检查点 ×9**（iter000-040,每 5 轮） | node09 `~/h20_validation_20260724/results/gomoku_ckpt_p15/` | 362 MB | ❌ 唯一副本,离线评估全靠它 |
| pure/rules 检查点（9×9 各 11 个） | node09 同目录 `gomoku_ckpt_{pure,rules}/` | 123 MB ×2 | ❌ 唯一副本 |
| 训练指标 JSON / GPU 日志 / 标定 jsonl | 本地 `results/` 与 node09 双份 | ~2.5 MB | 双份互备 |
| 网页权重导出（fp16 + manifest + 参考向量） | 本地 `results/web_export/` 与 node09 双份 | 21 MB | ✅ 可由 iter040.pt 重导 |
| 四份报告 | 本地 `report/` | 29 MB（play 占 28） | ✅ 全部可由 results/ + scripts/ 再生 |
| `gpu_log_gomoku_p15.csv` | 本地（**已合并** part1+part2+尾段,35,197 样本） | 1.0 MB | node09 上仍是三段,本地这份是权威 |

## 3. 任务账本

### 已完成（按时间）

- [x] 2026-07-24 通路验证 demo + 报告
- [x] 2026-07-25 9×9 双臂训练、臂间对打评估、`report/gomoku.html`（含测量分辨率修正:臂内梯子只排顺序等,见 CLAUDE.md）
- [x] 2026-07-31 Phase-0 三组标定;Phase-2 评估体系（AZPlayer 温度采样、锚点池、分级战术题+精确校验器、`run_gomoku15_in_container.sh`）;报告模板新卡与单臂容错;战术校验器"对手无成五点"加固（随机化对抗复核 1,794 声明 0 驳倒）
- [x] 2026-08-17 Phase-1 40 轮全程（含 2 次在线换配置的断点续跑、1 次温度干预）;`AZ_RESUME_ITER` 断点续跑;`report/gomoku15.html`;发射前多智能体审查修掉 5 处报告硬编码
- [x] 2026-08-18 人机对战页全链路（导出→WebGL2 推理→JS MCTS→对拍验证→交互验证）,抓修 GPU GroupNorm 单遍方差、纹理单元 clobber、aiTurn 回合守卫、执白悔棋死局等 9 个 bug

### Backlog（按优先级,均可独立开工）

1. **离线硬探针**：现役 10 题第 5 轮就三档全满——正解全在"最显眼的线"上,rusher 策略即可全对,没有区分"注意到线"和"算清强制序列"。做法:新题正解偏离显眼线（诱导长线与必防点分离;首个冲四不在最长线上的 VCF）,只读 `gomoku_ckpt_p15/` 逐检查点评测,模式仿 `eval_gomoku_cross_arm.py`（约 20 分钟容器任务）
2. **绝对强度基线换代**：pure-MCTS 在 15×15 全档饱和（含 8000 playouts,随机 rollout 在 225 格无估值能力）,正式退役;下一代用 VCF/VCT 求解器（`_vcf_starts` 已是雏形,需加防守方与深度扩展）或外部引擎
3. **`resumed_at` 一行修**：`train_rl_gomoku_alphazero.py` resume 块里 `M.get("resumed_at", [])` 应改为 `old.get("resumed_at", [])`,否则多次续跑只留最后一条
4. **Phase-3 吞吐工程**（要在 15×15 之上再上规模才需要）：playout cap randomization（预计 2-3×）、死和裁定、认输阈值、每卡单推理服务进程。实测依据见 CLAUDE.md Phase-0/Phase-1 记录
5. **Phase-4 冷启动 A/B v2**（如果还关心该问题）：把区分度做出来——每轮局数砍到 1/4 让先验值钱、判据改"到达固定强度的轮次"、≥3 种子测种子间方差;等 Phase-3 把单跑成本压下来再做
6. （小）人机对战页可选增强：AI 落子随机化（温度档,现在同局面必同手）、开局库、移动端触控优化

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
| 人机对战页（从检查点重导权重） | 容器内 `AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 CAL_CKPT=results/gomoku_ckpt_p15/iter040.pt python scripts/export_gomoku_web.py`,rsync 回 `results/web_export/` 再本地组装 |

## 5. git 托管建议

- **必须提交**：`scripts/`、`CLAUDE.md`、`STATUS.md`。这三样 + node09 上的检查点 = 一切可再生
- **建议提交**：`results/` 里的 JSON/JSONL/CSV（~2.5 MB,训练历史的唯一结构化记录,报告的输入）;`report/index.html`、`gomoku.html`、`gomoku15.html`（<1 MB,直接可看）
- **别直接提交**：`report/gomoku_play.html`（28 MB）与 `results/web_export/weights_fp16.bin`(21 MB)——要么走 Git LFS,要么按 §4 从 iter040.pt 再生;`data/`（CIFAR,一直不进 git）
- **检查点不在 git 里**：362+246 MB 且在 node09;如担心节点数据丢失,先 `rsync` 一份 `gomoku_ckpt_p15/` 到可靠存储再说

```gitignore
# 建议 .gitignore
data/
report/gomoku_play.html
results/web_export/weights_fp16.bin
results/**/*.pt
results/gomoku_ckpt_*/
__pycache__/
.DS_Store
```

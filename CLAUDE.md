# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

GPU 训练通路验证 demos：在远程 GPU 节点 **node09**（`node09.tx.bj.stonewise.cn`，8 × NVIDIA H20）上用 Docker 跑训练测试，并生成自包含的 HTML 可视化报告。目标是验证"环境通、能训练、可展示"，不是追求 SOTA 指标。

目前有两组互相独立的 demo，各有自己的容器入口脚本和报告：

| demo | 入口脚本 | 报告 | 内容 |
| --- | --- | --- | --- |
| 通路验证 | `run_all_in_container.sh` | `report/index.html` | ResNet-18 / CIFAR-10 + DQN / CartPole-v1 |
| AlphaZero 五子棋 | `run_gomoku_in_container.sh` + `run_gomoku_cross_arm.sh` | `report/gomoku.html` | 9×9 Gomoku，MCTS 自我对弈，两条臂 A/B |

## Two-machine workflow (the key thing to understand)

代码在本地 Mac 编写，训练在 node09 的 Docker 容器内执行，结果拷回本地生成报告：

1. **同步脚本到 node09**：`rsync -az scripts node09:~/h20_validation_20260724/`（SSH 别名 `node09` 已配好，用户 `fudong`）
2. **在容器内跑训练**（node09 上）：
   ```bash
   # 通路验证 demo（单卡）
   cd ~/h20_validation_20260724 && docker run --rm --gpus '"device=0"' \
     --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
     -v $PWD:/workspace -w /workspace \
     node09-h20-validation:20260717 bash scripts/run_all_in_container.sh

   # AlphaZero 五子棋（多卡 self-play，跑 ~3 小时，用 -d + --name 后台跑）
   cd ~/h20_validation_20260724 && docker run -d --name az_gomoku --gpus all \
     --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
     -v $PWD:/workspace -w /workspace \
     harbor.stonewise.cn/base/nvidia/pytorch:25.11-py3-cuda13.0-torch2.10 \
     bash scripts/run_gomoku_in_container.sh

   # 两条臂都跑完之后：从检查点做臂间对打（~22 分钟，不重新训练）
   cd ~/h20_validation_20260724 && docker run --rm --gpus all \
     --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
     -v $PWD:/workspace -w /workspace \
     harbor.stonewise.cn/base/nvidia/pytorch:25.11-py3-cuda13.0-torch2.10 \
     bash scripts/run_gomoku_cross_arm.sh
   ```
3. **取回结果**：`rsync -az node09:~/h20_validation_20260724/results/ results/`
4. **生成报告**：`python3 scripts/gen_report.py` → `report/index.html`；`python3 scripts/gen_gomoku_report.py` → `report/gomoku.html`（都是本地跑，无第三方依赖）

## node09 environment facts (verified 2026-07)

- Docker 默认 runtime 就是 nvidia；两个镜像都是现成的，**不要重新构建或 pull**：
  - `node09-h20-validation:20260717`（NGC PyTorch 24.08：torch 2.5 + CUDA 12.6 + torchvision + gymnasium + matplotlib）— 通路验证 demo 用
  - `harbor.stonewise.cn/base/nvidia/pytorch:25.11-py3-cuda13.0-torch2.10`（torch 2.10 + CUDA 13.0）— AlphaZero demo 用
- `/data_nvme0` 是 root-owned，普通用户不可写；工作目录用 `~/h20_validation_20260724`（即 `/home/fudong/...`）
- **网络**：CIFAR-10 官方源（cs.toronto.edu）只有 ~19 KB/s，不可用。数据走 **hf-mirror.com**（~8 MB/s）；pip 走清华镜像（tuna）也快。容器内 pip 默认源可用但慢
- 容器内写入挂载卷的文件 owner 是 root，本地 rsync 回来正常，但在 node09 上二次修改/删除需要开一个一次性容器进去做

## Pipeline structure

### 通路验证 demo

`scripts/run_all_in_container.sh` 是容器内的总入口，依次执行：

- `prep_cifar10.py` — 把 hf-mirror 下载的 parquet 转成单个 `data/cifar10.npz`（parquet 需预先 curl 到 `data/`，见 run_all 脚本；`data/` 不在 git 里）
- `train_dl_cifar10.py` — 单卡 bf16 AMP 训练，写 `results/dl_metrics.json`（per-step loss + per-epoch acc/吞吐）
- `train_rl_cartpole.py` — DQN，写 `results/rl_metrics.json`，其中 `trajectories` 字段记录了 6 个检查点的贪心策略状态序列（小车位置+杆角），供报告里的 Canvas 动画回放用
- `gpu_monitor.sh start|stop` — 后台逐秒采样 nvidia-smi 到 CSV，DL/RL 阶段各存一份（`gpu_log_dl.csv` / `gpu_log_rl.csv`）

两个训练脚本各自独立、无共享代码；改指标 schema 时要同步改 `gen_report.py` 和 `report_template.html`。

### AlphaZero 五子棋 demo

`scripts/run_gomoku_in_container.sh` 把 `train_rl_gomoku_alphazero.py` **跑两遍**，同种子、同配置，唯一差别是 `AZ_BETA0`：

- arm `pure`（`AZ_BETA0=0`）— 标准 AlphaZero，纯 self-play
- arm `rules`（`AZ_BETA0=0.6`）— 冷启动阶段在 MCTS 展开时把战术规则先验混进网络先验（`P = (1-β)·P_net + β·P_rule`），β 在前 `AZ_RULE_ITERS=20` 轮线性退火到 0，之后两条臂算法完全相同。训练目标不变，评估一律强制 β=0

这两条臂就是报告里"训练效率对比"一节的数据来源——回答"冷启动先用规则能不能更快"。每条臂各写一份 `results/gomoku_{tag}.json` + `gpu_log_gomoku_{tag}.csv` + `gomoku_{tag}.log`，检查点在 `results/gomoku_ckpt_{tag}/`。

`train_rl_gomoku_alphazero.py` 的要点（全部通过 `AZ_*` 环境变量配置，默认值见 run 脚本）：

- **批量 lockstep MCTS**：几百局棋同时推进，每步模拟合成一次批量前向；树遍历是 Python/CPU-bound，所以 self-play 用多进程（默认 8 个 worker，按 `AZ_GPUS=0,0,1,1,2,2,3,3` 分到 4 张卡），梯度更新单独在 `AZ_DEVICE=cuda:0`
- **网络**：4 输入平面 → 128 通道 stem → 8 残差块，**GroupNorm 而不是 BatchNorm**（self-play 的批量随着对局陆续结束而不断缩小，GroupNorm 在 train/eval 下行为一致，BN 的 running stats 会不可靠）
- 采样时做 8 种二面体对称增强；`AZ_SEED` 固定
- **评估**：每 `AZ_EVAL_EVERY=5` 轮对三档固定基线打一批（random / rule-greedy 贪心启发式 / pure-MCTS 400 与 1000 playouts），另加 4 个有唯一正解的战术测试位（成五、封堵成五、堵活三、活三扩张），分别记录网络原始策略和加 MCTS 后的落点。**pure-MCTS 会很快被打到 100%，rule-greedy 才是有区分度的那条基线**
- 训练结束跑检查点循环赛，用 Bradley-Terry MLE 拟合 Elo（初始权重 = 0 基准）。**2026-07-31 起循环赛和锚点对局的落子带温度采样**（`AZ_ELO_TEMP`/`AZ_ANCHOR_TEMP`，默认 0.3），每局互不相同，Elo 有真实分辨率（约 `347/√((检查点数-1)×每对局数)`）；温度设为 0 就回到旧的确定性行为——那时每对 `AZ_ELO_GAMES` 局里只有 2 局互不相同，梯子只能排顺序（老的 pure/rules JSON 就是这么跑的，报告模板按 `elo_detail.temp` 自动切换措辞）

### Phase-2 评估体系（2026-07-31 加入 trainer，Phase-1 长跑前完成）

上一趟 50 轮的教训是"第 15 轮之后所有仪表恒定"。为此 trainer 加了三样东西，全部向后兼容（老 JSON 里没有对应字段，模板会把相应卡片 `remove()` 掉）：

- **`AZPlayer` 温度采样**：`temp>0` 时按访问数^(1/τ) 采样且尊重传入的 rng（旧行为是 argmax + 忽略 rng）。评估用小温度（0.3）目的是让对局互不相同,不是削弱棋力
- **锚点池滚动对局**（`M["anchors"]`）：每次 eval 时当前网络对最近 `AZ_ANCHOR_K=3` 个检查点各打 `AZ_ANCHOR_GAMES=12` 局带温度对局。对最新锚点得分 >50% = 还在变强，**这把尺子永远不会饱和**，是长跑中判断"还要不要继续训"的主仪表
- **分级战术题**（`tactical_positions()` 重写）：不再手写坐标，改为"模式构造 + 精确校验器"（`_win_cells`/`_double_threat_moves`/`_vcf_starts`/`_defense_moves`/`_block_five_moves`），任意 `AZ_BOARD>=9` 可用，每题 good 集合由校验器计算而非人工断言。共 10 题 = 5 模式 × 2 方向：T1 成五/封五（1 手）、T2 活四攻防（3 手内强制）、T3 连续冲四 VCF（≥2 手深）。四三/双活三需要 VCT 搜索,不在覆盖范围。`tactical[].tier_acc` 按档记正确率；本地无 torch 也能测试这部分（纯 numpy,见生成器自带的 assert）；selftest 只要求规则先验解出 T1。**校验器把"对手无成五点"作为冲四/双威胁成立的前提**（2026-07-31 随机化对抗复核加固：没这个检查时,随机带威胁局面上 372/949 的 VCF 声明和 84/3037 的双威胁声明是错的——对手可以不应四直接成五）
- **Phase-1 入口**：`run_gomoku15_in_container.sh`（15×15/192ch/12blk/800sims/28workers/1344局/40轮,单臂 `p15`,`AZ_PURE_PLAYOUTS_HARD=8000`），预计 ~14-16 分钟/轮、一夜跑完。冒烟验证过的小配置在该脚本注释和 git 历史里

### Phase-1 实跑记录（2026-08-17，40 轮完成 → `report/gomoku15.html`）

**结果**：终局循环赛 Elo +1,639（温度采样,±55）,三档 pure-MCTS（400/1000/8000）与 rule-greedy 全部 12-0/8-0,总耗时 10.7 h（含两次拓扑切换的空档）。要点全在过程里：

- **短对局塌缩是 15×15 的真实陷阱**：手数从 58 塌到 11-14、黑胜 96%、零和棋,持续了第 5-25 轮。原因不是网络不会防守（argmax 评估防守正常、rule-greedy 4-4）,而是 `AZ_TEMP_MOVES=20 > 对局长度`——全程温度采样让白棋每次强制防守都有失手概率,黑棋靠廉价速胜刷数据。**锚点梯子是唯一及时报警的仪表**：5 轮间隔得分 92%→67%→58%→50%（停滞）;固定基线和战术题那时早已饱和
- **干预与验证**：iter 25 把 `AZ_TEMP_MOVES` 20→10 续跑,3 轮内手数 13→24、白胜 3%→15%、出现 225 手和棋、value 头 ev 从 0.89 回落 0.71（重新有非平凡预测任务）;锚点 5 轮间隔得分回到 75%→83%,循环赛里 iter25→30 一段 +386 Elo（此前 10→25 十五轮才 +157）。**iter 20 那次纯拓扑续跑（buffer 同样清空）没有带来恢复,归因干净**
- **trainer 有断点续跑**（`AZ_RESUME_ITER=N`）:加载 `iterNNN.pt`、接续 JSON 历史、重建 snapshot 列表、重放 LR 调度;buffer 和优化器动量不持久化。已知小 bug:`resumed_at` 合并时从新 M 读取,多次续跑只留最后一条（改成从 old 读即可）。**`gpu_monitor.sh start` 会截断 CSV**——重启前先 `cp` 出 partN,报告前按时间戳合并
- **吞吐实测**：28w×48g/卡均 4 = 全卡 100% util,~19-22 min/轮（标定外推的 2 倍慢:15×15 标定只测了每卡 1 进程,大网络下每卡 4 进程的 contention 远超 9×9 的 30%）;12w（3 卡×4,每 worker 112 局）≈ 10-11 min/轮@手数13、~20 min/轮@手数 20——**batch 效应补回了大半 worker 损失,少卡大 batch 在这个网络规模下性价比更高**
- **评估体系的 15×15 现实**：pure-MCTS 全档（含 8000 playouts）从第 5 轮起 12-0,随机 rollout 在 225 格上没有估值能力,**这条基线可以退役**;分级战术题 iter 5 起三档全满——正解都落在"最显眼的线"上,rusher 策略即可全对,**分档没有把"注意到线"和"算清强制序列"区分开**。待办:离线硬探针（正解偏离显眼线:诱导线与必防点分离、首个冲四不在最长线上的 VCF）,只读检查点,模式同臂间对打。绝对强度基线下一步得用 VCF/VCT 求解器或外部引擎
- **风格相克现象**：iter020（冲锋流巅峰）对 iter 25/35 的带温度对局意外顽强（两次 6-6）,而 iter 30/40 对 25/30 都是 9-3——读终局梯子时先想到非传递性,别把单点当"没进步"

### 臂间对打（`eval_gomoku_cross_arm.py`）

固定基线和战术题在 15 轮内就全部饱和（rule-greedy 从第 15 轮起 8-0，pure-mcts-400 从第 5 轮起 12-0，战术题第 5 轮起 4/4），**饱和之后它们完全区分不出两条臂**。所以另有一个只读检查点、不重新训练的评估，由 `scripts/run_gomoku_cross_arm.sh` 在两条臂都跑完之后调用，写 `results/gomoku_cross_arm.json`：

- **同轮对打**：每个共有轮次上 `pure@k` vs `rules@k`，永远不会饱和，是"同等算力下谁更强"的直接测量。得分率从 rules 的视角算，> 0.5 = rules 领先
- **独立观测的单位是"开局"，不是"局"**。臂间对打里 `AZPlayer` 用默认 `temp=0`：取访问数 argmax、根节点不加噪声、忽略传进去的 rng（2026-07-31 起 `temp>0` 才会采样），所以同一份权重在同一个局面下每次都走同一步：从空棋盘开打 N 局，其实只是 2 局（各一种先后手）复制 N/2 遍，按 N 算标准误会把精度夸大 √(N/2) 倍。所以 `play_pair()` 自己写 lockstep 循环（不能用 `az.play_matches`），先随机 `AZX_OPEN_PLIES=2` 手开局、每个开局双方各先手一次，**一个开局的两局合起来算一个观测**，写进 `pair_scores`（取值 {0, .25, .5, .75, 1}）。报告的误差条就是对 `pair_scores` 求标准误
- **联合 Elo**：把两条臂的检查点放进同一个循环赛做一次 Bradley-Terry 拟合，于是所有检查点在同一把尺子上；再把 rules 的 Elo 读到 pure 的 Elo-vs-轮次曲线上，就把"更强"换算成"省下 N 轮自我对弈"（`interp_iter()` 用 running max 保证换算对目标值单调）
- 第 0 轮两条臂是同一份随机初始权重（同种子），`same_file()` 检出后两项测量都跳过它——自己打自己没有信息量
- 它 `import train_rl_gomoku_alphazero`，而 `AZNet()` / `State()` 读的是那个模块的**模块级全局变量**（来自 `AZ_*` 环境变量）。棋盘尺寸对不上会静默地在错误棋盘上对局，所以 `check_config()` 会拿每条臂 JSON 里记录的配置比对，不一致就直接退出——改 run 脚本里的 `AZ_BOARD/AZ_CH/AZ_BLOCKS/AZ_NIR` 时两个 run 脚本要一起改
- 每对独立，所以按 `AZX_GPUS` × `AZX_PROCS_PER_GPU` farm 到多进程（一进程一个 CUDA context）
- **实测耗时**（9×9 / 128ch / 8 blocks）：同轮对打一对 16 开局 × 400 模拟约 115 s；Elo 一对 6 开局 × 200 模拟，两个强度接近的检查点约 80 s（一边倒的对局 10 s 内就结束）。11 + 10 个检查点 = 210 对 Elo，所以总时间是 Elo 主导，14 个 worker 上约 22 分钟。GPU 5 上有别人的任务，`AZX_GPUS` 默认避开它

**这一趟跑出来的答案（2026-07-25，50 轮 × 两条臂）：在这个规模上测不出规则冷启动的优势。** 全程 160 个开局对 / 320 局，rules 得分率 50.9% ± 3.9pp（2 SE）；退火区间内（≤20 轮）53.9% ± 7.7pp，退火之后 49.0% ± 4.0pp，两段之差也在噪声里；联合 Elo 峰值 pure +1737 / rules +1749，逐轮差值在 −114…+94 之间摆动而拟合分辨率是 ±32；"省下的轮次"中位数 −3 轮而换算误差 ±22 轮。规则先验也不额外花时间（50 轮多 1.7 分钟 / 1.5%）。**要在这个题目上真做出差别，得把区分度做出来**：更大的棋盘、更少的每轮 self-play 局数（让早期数据更稀缺、先验更值钱），或者只比"到达某个固定强度用了几轮"而那个强度不能在 15 轮内饱和。

### Phase-0 标定（2026-07-31，`run_phase0_calib.sh` → `results/calib_*.jsonl`）

为规划下一轮训练跑的三组标定，`calib_draw_vs_sims.py` / `calib_selfplay_point.py` 都只 import trainer 不改它：

- **9×9 是棋本身和棋，不是搜索太浅**：iter050 权重、2 手随机开局、argmax 无噪声自对弈，和棋率在 400/1600/6400 sims 下分别 79.7% / 81.2% / 75.0%（合并 127/160 = 79% ± 6.4pp），搜索加深 16 倍纹丝不动。**在 9×9 上继续加算力没有意义，加大棋盘才有**。白棋在 160 局里只赢了 2 局
- **self-play 吞吐**（9×9/128ch/400sims，positions/s）：8w×48g 基线 160 → 8w×192g 228（batch 效应 1.43×）→ 28w×48g 417（worker 效应 2.61×，每卡 4 进程约有 30% 互相拖慢）→ 28w×192g 727（**4.55×**，batch 效应在高 worker 数下更强 1.74×）。瓶颈主要是 Python 树遍历（CPU 随局数线性），GPU batch 摊薄只是次要项——**同样总局数下，多 worker 优于大 batch**
- **15×15/192ch/12blk/800sims 实测**（随机初始权重）：4w×48g = 23.8 positions/s，4w×192g = 35.4（batch 1.49×）；外推 28 worker 约 110-160 positions/s。一轮 5,376 局在这个配置下要 40+ 分钟，**过夜预算只装得下每轮 ~1,344 局（28w×48g）× 40-50 轮**；要更多轮次得先做 playout cap randomization / 认输 / 死和裁定这类吞吐工程。随机权重下 960 局零和棋（和棋是学出来的防守的产物，大棋盘前期训练信号很干净）

### 人机对战页（2026-08-18，`report/gomoku_play.html`，28 MB 单文件）

用 iter040 权重做的浏览器内人机对战,两步构建：

1. **导出**（node09 容器,唯一要 torch 的一步）：`export_gomoku_web.py` → `results/web_export/`（fp16 权重二进制 + manifest + 5 个参考向量）。参考输出用"fp16 取整后的权重 + fp32 计算"生成——和浏览器的数值模型一致,`empty` 盘那条曾抓过真 bug,别删
2. **本地组装**：`gen_gomoku_play.py` 把三个文件内嵌进 `gomoku_play_template.html`。页面开机自测:5 向量对拍,徽章显示 maxΔ（正常 ≤2e-6）,失败自动回退 CPU 引擎

要点/坑（都是真踩过的）：

- **推理**：trunk 在 WebGL2（R32F 纹理,192 通道摊成 16×12 tile 网格;conv3x3 / GroupNorm 两遍统计 / apply 三个 shader）,heads 读回 CPU 算。**GN 统计必须两遍**（E[x²]−E[x]² 在近常量激活下 fp32 灾难性消去——空棋盘就触发）;**所有权重纹理必须初始化时预建**——`gl.bindTexture` 挂在当前活跃纹理单元上,渲染中途懒创建会把已绑的激活纹理顶掉,症状是"只有第一次前向错"
- **MCTS**：逐语义复刻 trainer 的 `Tree`（PUCT c=3、终局值 mover 视角、backup 逐层翻号——根节点 W/N 已是根方视角,**不要再取负**）。UI 的 `aiTurn` 必须校验"轮到 AI"（悔棋/驱动脚本会制造不是 AI 回合的调用）;执白悔棋可能回滚到 AI 先手局面,悔棋路径要主动re-trigger `aiTurn`
- **无头验证**：`--use-angle=swiftshader` 下 WebGL 可用但慢 ~50 倍,驱动对局用 sims=0/32,否则 3 手 128 sims 就超 7 分钟（审查 agent 全卡死过一回）;**无头老模式窗口宽度下限 500px**,`--window-size=390` 拍出来的"溢出"是伪影,量 `innerWidth` 确认;DPR 路径用 `--force-device-scale-factor=2` 测
- 移动端:grid 轨道要 `minmax(0,1fr)` + item `min-width:0`,否则 canvas 固有宽度撑破单列布局

## Report generation

两个报告都是同一套做法：`gen_*.py` 读 `results/*` 并把 JSON 内联进对应模板的 `__DATA_JSON__` 占位符，产出零依赖、可直接分发的单个 HTML。

- `gen_report.py` + `report_template.html` → `report/index.html`
- `gen_gomoku_report.py` + `gomoku_report_template.html` → `report/gomoku.html`（`GOMOKU_RESULTS` / `GOMOKU_OUT` 可覆盖输入输出路径，方便用假数据先跑通模板；`GOMOKU_ARMS` 指定臂列表，默认 `pure,rules`，单臂如 `GOMOKU_ARMS=p15` 时整个 A/B 对比节自动消失、"两条臂并列"标签改成"单臂训练"）。锚点得分率 / 分档战术两张新卡只在数据里有 `anchors` / `tier_acc` 字段时出现，老 JSON 照常渲染

模板是手写 SVG/Canvas 图表（无图表库），遵循 dataviz skill 的规范：CSS 变量定义浅色/深色两套配色（已通过 palette 验证器）、每张图带悬浮 tooltip 和表格视图、文本一律 `textContent`（不用 innerHTML）。改完报告要用 headless Chrome 截图检查浅色和深色两种模式。

**截图时两套配色都要显式 pin `data-theme`**：深色规则同时挂在 `@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) }` 和 `:root[data-theme="dark"]` 上，而 headless Chrome 不报告任何 color-scheme 偏好——只给深色那张加 `data-theme="dark"`、浅色那张不加，两张会渲染成一模一样的浅色图，看起来像"深色模式没生效"。浅色那张也写 `data-theme="light"`，然后**比对两张 PNG 的 md5 确认它们真的不同**再去看内容。

改 gomoku 报告时注意两点：

- GPU 曲线走 `bucket_gpu()` 做等长时间窗**均值**降采样，不是抽稀——AlphaZero 的 self-play/train 占空比在 1 Hz 原始数据上抽稀后是一片噪声
- 训练要跑几个小时，改模板别等真数据。用假数据先把整条链路跑通再对真数据生成（本次就是这么发现棋盘尺寸、胜利连线高亮、阈值表非单调这几个 bug 的）

模板对缺数据是容错的，两种情况都验证过：没有 `gomoku_cross_arm.json` 时整个"①两条臂直接对局"块被 `remove()` 掉、"②"编号也跟着去掉；拿一条还在训练中的臂生成时（`elo` 字段是训练最后才写的）Elo 卡片和 tile 一起消失，不会画出空坐标轴。

`gen_gomoku_report.py` 会把每条臂的 `elo_detail.pairs`（循环赛逐对结果，只用于事后审计）丢掉，但**保留 `sims` / `games_per_pair` / `seconds`**——模板要用 `games_per_pair` 在 Elo 卡片下写清"每对 N 局里只有 2 局互不相同"。

棋谱回放的取样是**先按轮次去重、再抽稀轮次**（`by_iter` → `thin(sorted(by_iter), 8)`），不是直接对 `sample_games` 列表抽稀：训练器每轮存两局，直接切局会只留下前四轮、回放标签停在 iter 19 而全程有 50 轮——页面看着正常，实际把后半段训练悄悄藏掉了。评估棋谱另算：重复出现的那个对手抽 3 局，加上 `rest[-2:]` 保证只出现一次的最终强基线一定在里面。

**结论的措辞要跟着测量的分辨率走。** 页面上有三层强度证据，越靠前越不该拿去回答 A/B 问题：固定基线和战术题（15 轮内饱和，区分不了两条臂）、臂内检查点循环赛 Elo（确定性对局，只排顺序）、臂间随机开局对打和联合 Elo（唯一有误差条的）。顶部结论段和"直接对局怎么读"都优先用最后一层，只在没有臂间数据时才退回前面几层，并且退回时明说"只能看个大概"。同理，"省下 N 轮"这一列自带换算误差（Elo 拟合分辨率 ÷ 曲线后半段斜率），中位数小于该误差时模板会改口说"读不出确定的轮次差"而不是照报数字。

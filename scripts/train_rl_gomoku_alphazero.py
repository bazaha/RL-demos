"""AlphaZero on Gomoku (five-in-a-row), trained from scratch on node09's H20s.

Self-play + MCTS + policy/value network. No human games, no opening book.

Two things here are worth knowing about before reading the code:

1.  MCTS runs for many games in lockstep, so every simulation step becomes one
    batched network forward pass. A tree search in Python is otherwise
    latency-bound and leaves the GPU idle.

2.  Optional rule-guided cold start (AZ_BETA0 > 0). Early on, a cheap tactical
    oracle is mixed into the search prior at every expansion:

        P = (1 - beta) * P_net + beta * P_rule,   beta annealed to 0

    The training target stays the MCTS visit count, so the rules only steer
    exploration -- once beta hits 0 this is plain AlphaZero. Evaluation always
    runs at beta = 0, which is what makes the A/B against AZ_BETA0=0 fair.

Self-play is farmed out to worker processes (one CUDA context each, several per
GPU) because the tree walk is CPU-bound; training stays in the parent on GPU 0.

Writes results/<AZ_OUT>.json:
  iterations[]   per-iteration losses, self-play stats, timings, beta
  evals[]        win rate vs random / pure-MCTS baselines at checkpoints
  tactical[]     policy+MCTS answers on hand-built tactical positions
  sample_games[] full move lists for board replay animation
  elo[]          Bradley-Terry Elo of checkpoints from a round-robin
  phases[]       (kind, t0, t1) spans so the GPU log can be shaded
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


def _ei(k, d):
    return int(os.environ.get(k, d))


def _ef(k, d):
    return float(os.environ.get(k, d))


BOARD = _ei("AZ_BOARD", 9)
N_IN_ROW = _ei("AZ_NIR", 5)
N_ACT = BOARD * BOARD

CHANNELS = _ei("AZ_CH", 128)
BLOCKS = _ei("AZ_BLOCKS", 8)

ITERS = _ei("AZ_ITERS", 50)
GAMES_PER_ITER = _ei("AZ_GAMES", 256)
N_SIMS = _ei("AZ_SIMS", 400)
C_PUCT = _ef("AZ_CPUCT", 3.0)
DIR_ALPHA = _ef("AZ_DIR_ALPHA", 0.3)
DIR_EPS = _ef("AZ_DIR_EPS", 0.25)
TEMP_MOVES = _ei("AZ_TEMP_MOVES", 12)

# rule-guided cold start: beta = BETA0 * max(0, 1 - (it-1)/RULE_ITERS)
BETA0 = _ef("AZ_BETA0", 0.0)
RULE_ITERS = _ei("AZ_RULE_ITERS", 20)

BUFFER_CAP = _ei("AZ_BUFFER", 200_000)
TRAIN_STEPS = _ei("AZ_TRAIN_STEPS", 500)
BATCH = _ei("AZ_BATCH", 512)
LR = _ef("AZ_LR", 2e-3)
WD = _ef("AZ_WD", 1e-4)

EVAL_EVERY = _ei("AZ_EVAL_EVERY", 5)
EVAL_SIMS = _ei("AZ_EVAL_SIMS", 400)
EVAL_GAMES_MCTS = _ei("AZ_EVAL_GAMES", 12)
EVAL_GAMES_RAND = _ei("AZ_EVAL_GAMES_RAND", 8)
PURE_PLAYOUTS = _ei("AZ_PURE_PLAYOUTS", 400)
PURE_PLAYOUTS_HARD = _ei("AZ_PURE_PLAYOUTS_HARD", 1000)
ELO_SIMS = _ei("AZ_ELO_SIMS", 200)
ELO_GAMES = _ei("AZ_ELO_GAMES", 4)
ELO_TEMP = _ef("AZ_ELO_TEMP", 0.3)

# rolling anchor matches: at each eval, the current net plays the most recent
# ANCHOR_K checkpoints with a small temperature. Fixed baselines saturate a few
# iterations in; this ladder never does.
ANCHOR_GAMES = _ei("AZ_ANCHOR_GAMES", 12)
ANCHOR_K = _ei("AZ_ANCHOR_K", 3)
ANCHOR_SIMS = _ei("AZ_ANCHOR_SIMS", 200)
ANCHOR_TEMP = _ef("AZ_ANCHOR_TEMP", 0.3)

SEED = _ei("AZ_SEED", 42)
# resume from the checkpoint saved at this eval iteration: loads
# CKPT_DIR/iterNNN.pt + the metrics JSON at AZ_OUT, rebuilds the snapshot list
# from disk, and continues the loop at NNN+1. The replay buffer and optimizer
# moments are NOT persisted, so the first couple of resumed iterations train
# on fresh self-play only -- acceptable for a topology change mid-run.
RESUME_ITER = _ei("AZ_RESUME_ITER", 0)
DEVICE = os.environ.get("AZ_DEVICE", "cuda:0")
GPUS = [int(g) for g in os.environ.get("AZ_GPUS", "0").split(",") if g != ""]
OUT_JSON = os.environ.get("AZ_OUT", "results/gomoku_metrics.json")
CKPT_DIR = os.environ.get("AZ_CKPT", "results/gomoku_ckpt")
TAG = os.environ.get("AZ_TAG", "az")

DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


# --------------------------------------------------------------------------- #
# game rules
# --------------------------------------------------------------------------- #
def _wins(board, r, c, p):
    """True if the stone just placed at (r, c) by p completes N_IN_ROW."""
    for dr, dc in DIRS:
        cnt = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < BOARD and 0 <= cc < BOARD and board[rr, cc] == p:
            cnt += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < BOARD and 0 <= cc < BOARD and board[rr, cc] == p:
            cnt += 1
            rr -= dr
            cc -= dc
        if cnt >= N_IN_ROW:
            return True
    return False


class State:
    """Gomoku position. board holds +1 / -1 / 0; to_play is +1 (black) or -1."""

    __slots__ = ("board", "to_play", "last_move", "n_moves", "winner", "done")

    def __init__(self):
        self.board = np.zeros((BOARD, BOARD), dtype=np.int8)
        self.to_play = 1
        self.last_move = -1
        self.n_moves = 0
        self.winner = 0
        self.done = False

    def clone(self):
        s = State.__new__(State)
        s.board = self.board.copy()
        s.to_play = self.to_play
        s.last_move = self.last_move
        s.n_moves = self.n_moves
        s.winner = self.winner
        s.done = self.done
        return s

    def legal_mask(self):
        return self.board.reshape(-1) == 0

    def play(self, a):
        r, c = divmod(a, BOARD)
        p = self.to_play
        self.board[r, c] = p
        self.last_move = a
        self.n_moves += 1
        if _wins(self.board, r, c, p):
            self.winner = p
            self.done = True
        elif self.n_moves == N_ACT:
            self.winner = 0
            self.done = True
        self.to_play = -p

    def encode(self):
        """4 x BOARD x BOARD planes, always from the mover's point of view."""
        x = np.zeros((4, BOARD, BOARD), dtype=np.float32)
        tp = self.to_play
        x[0] = self.board == tp
        x[1] = self.board == -tp
        if self.last_move >= 0:
            x[2, self.last_move // BOARD, self.last_move % BOARD] = 1.0
        if tp == 1:
            x[3] = 1.0
        return x

    def terminal_value(self):
        """Game result from the point of view of the player who must move now."""
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == self.to_play else -1.0


# --------------------------------------------------------------------------- #
# tactical rule oracle, vectorised over a batch of positions
#
# Cost is ~60 numpy ops regardless of batch size, so calling it once per
# batched simulation step is cheap enough to guide every expansion.
# --------------------------------------------------------------------------- #
_CENTER_W = None


def _center_weight():
    global _CENTER_W
    if _CENTER_W is None:
        c = (BOARD - 1) / 2.0
        rr, cc = np.meshgrid(np.arange(BOARD), np.arange(BOARD), indexing="ij")
        d = np.maximum(np.abs(rr - c), np.abs(cc - c))
        _CENTER_W = (1.0 / (1.0 + d)).astype(np.float32)
    return _CENTER_W


def _shift(a, dr, dc):
    """b[..., r, c] = a[..., r+dr, c+dc], zero/False outside the board."""
    b = np.zeros_like(a)
    r0, r1 = max(0, -dr), min(BOARD, BOARD - dr)
    c0, c1 = max(0, -dc), min(BOARD, BOARD - dc)
    if r0 < r1 and c0 < c1:
        b[..., r0:r1, c0:c1] = a[..., r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return b


def _max_run(own):
    """own: (B,H,W) bool. -> (B,H,W) int16 max line length if that cell is filled."""
    best = np.ones(own.shape, dtype=np.int16)
    for dr, dc in DIRS:
        tot = np.ones(own.shape, dtype=np.int16)
        for sign in (1, -1):
            m = np.ones(own.shape, dtype=bool)
            for k in range(1, N_IN_ROW):
                m = m & _shift(own, sign * dr * k, sign * dc * k)
                tot += m
        np.maximum(best, tot, out=best)
    return best


def rule_priors_batch(boards, to_play):
    """Hand-written tactical prior. boards (B,H,W) int8, to_play (B,) -> (B,N_ACT).

    Priority: complete five > block opponent's five > build/deny lines near the
    action. Deliberately shallow -- it knows nothing about double threats -- so
    it is a plausible "cold start heuristic", not a strong engine.
    """
    tp = to_play.reshape(-1, 1, 1)
    me = boards == tp
    opp = boards == -tp
    empty = boards == 0
    my_run = _max_run(me)
    opp_run = _max_run(opp)

    occ = boards != 0
    near = np.zeros(boards.shape, dtype=bool)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            near |= _shift(occ, dr, dc)
    # an empty board has no "near" cells; let the centre bias carry it
    near |= ~occ.any(axis=(1, 2))[:, None, None]

    base = (my_run.astype(np.float32) ** 2 + 0.9 * opp_run.astype(np.float32) ** 2)
    base = base * near + 0.5 * _center_weight()
    base = base * empty + 1e-6 * empty

    win_now = (my_run >= N_IN_ROW) & empty
    block_now = (opp_run >= N_IN_ROW) & empty
    has_win = win_now.any(axis=(1, 2))[:, None, None]
    has_block = block_now.any(axis=(1, 2))[:, None, None]

    p = np.where(has_win, win_now.astype(np.float32),
                 np.where(has_block, block_now.astype(np.float32), base))
    p = p.reshape(len(boards), N_ACT)
    s = p.sum(axis=1, keepdims=True)
    bad = s[:, 0] <= 1e-12
    if bad.any():  # fully blocked row: fall back to uniform over legal moves
        lm = empty.reshape(len(boards), N_ACT).astype(np.float32)[bad]
        p[bad] = lm / np.maximum(lm.sum(axis=1, keepdims=True), 1.0)
        s[bad] = 1.0
    return p / s


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)), inplace=True)
        y = self.n2(self.c2(y))
        return F.relu(x + y, inplace=True)


class AZNet(nn.Module):
    """Small AlphaZero-style trunk with policy and value heads.

    GroupNorm rather than BatchNorm: self-play evaluates the net on batches that
    shrink as games finish, and GroupNorm behaves identically in train and eval
    so there are no running-stats artefacts to chase.
    """

    def __init__(self, ch=CHANNELS, blocks=BLOCKS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch), nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.ph = nn.Sequential(
            nn.Conv2d(ch, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(32 * N_ACT, N_ACT),
        )
        self.vh = nn.Sequential(
            nn.Conv2d(ch, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(32 * N_ACT, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1), nn.Tanh(),
        )

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.ph(h), self.vh(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# MCTS
# --------------------------------------------------------------------------- #
class Node:
    """Edge statistics live in the parent as flat arrays, AlphaZero style."""

    __slots__ = ("state", "P", "N", "W", "children", "legal", "expanded")

    def __init__(self, state):
        self.state = state
        self.P = None
        self.N = None
        self.W = None
        self.children = {}
        self.legal = None
        self.expanded = False


class Tree:
    def __init__(self, state, c_puct=C_PUCT):
        self.root = Node(state)
        self.c_puct = c_puct

    def select(self):
        """Walk down by PUCT. Returns (path, leaf, terminal_value_or_None)."""
        node = self.root
        path = []
        while True:
            st = node.state
            if st.done:
                return path, node, st.terminal_value()
            if not node.expanded:
                return path, node, None
            N, W, P = node.N, node.W, node.P
            q = np.where(N > 0, W / np.maximum(N, 1.0), 0.0)
            u = self.c_puct * P * math.sqrt(N.sum() + 1.0) / (1.0 + N)
            score = q + u
            score[~node.legal] = -1e9
            a = int(score.argmax())
            path.append((node, a))
            child = node.children.get(a)
            if child is None:
                ns = st.clone()
                ns.play(a)
                child = Node(ns)
                node.children[a] = child
            node = child

    def expand(self, node, probs):
        legal = node.state.legal_mask()
        p = probs.astype(np.float64) * legal
        s = p.sum()
        p = p / s if s > 1e-12 else legal / legal.sum()
        node.P = p.astype(np.float32)
        node.N = np.zeros(N_ACT, dtype=np.float32)
        node.W = np.zeros(N_ACT, dtype=np.float32)
        node.legal = legal
        node.expanded = True

    @staticmethod
    def backup(path, leaf_value):
        """leaf_value is from the leaf mover's view; flip sign each level up."""
        v = leaf_value
        for node, a in reversed(path):
            v = -v
            node.N[a] += 1.0
            node.W[a] += v

    def apply_noise(self, rng, alpha, eps):
        node = self.root
        idx = np.flatnonzero(node.legal)
        if idx.size == 0:
            return
        noise = rng.dirichlet([alpha] * idx.size)
        p = node.P.copy()
        p[idx] = (1.0 - eps) * p[idx] + eps * noise.astype(np.float32)
        node.P = p

    def advance(self, a):
        child = self.root.children.get(a)
        if child is None:
            ns = self.root.state.clone()
            ns.play(a)
            child = Node(ns)
        self.root = child


def run_sims(net, trees, n_sims, device, noise_rng=None, beta=0.0):
    """n_sims PUCT simulations across all trees, batching every NN evaluation.

    beta > 0 mixes the tactical rule prior into the network prior at expansion.
    """
    if not trees:
        return
    n = len(trees)
    xbuf = np.empty((n, 4, BOARD, BOARD), dtype=np.float32)
    bbuf = np.empty((n, BOARD, BOARD), dtype=np.int8)
    tbuf = np.empty((n,), dtype=np.int8)
    for _ in range(n_sims):
        reqs = []
        for t in trees:
            path, leaf, tv = t.select()
            if tv is not None:
                t.backup(path, tv)
            else:
                reqs.append((t, path, leaf))
        if not reqs:
            continue
        k = len(reqs)
        for i, (_, _, leaf) in enumerate(reqs):
            xbuf[i] = leaf.state.encode()
            if beta > 0.0:
                bbuf[i] = leaf.state.board
                tbuf[i] = leaf.state.to_play
        with torch.no_grad():
            x = torch.from_numpy(xbuf[:k]).to(device, non_blocking=True)
            logits, values = net(x)
            probs = torch.softmax(logits, dim=1).float().cpu().numpy()
            vals = values.float().cpu().numpy()
        if beta > 0.0:
            rp = rule_priors_batch(bbuf[:k], tbuf[:k])
            probs = (1.0 - beta) * probs + beta * rp
        for i, (t, path, leaf) in enumerate(reqs):
            is_root = leaf is t.root
            t.expand(leaf, probs[i])
            if is_root and noise_rng is not None:
                t.apply_noise(noise_rng, DIR_ALPHA, DIR_EPS)
            t.backup(path, float(vals[i]))


# --------------------------------------------------------------------------- #
# self-play
# --------------------------------------------------------------------------- #
def selfplay(net, n_games, n_sims, device, rng, temp_moves=TEMP_MOVES, beta=0.0):
    """Play n_games in lockstep. Returns (X, PI, Z, move_logs, winners, lengths)."""
    net.eval()
    trees = [Tree(State()) for _ in range(n_games)]
    recs = [[] for _ in range(n_games)]
    logs = [[] for _ in range(n_games)]
    active = list(range(n_games))
    move_i = 0
    while active:
        run_sims(net, [trees[i] for i in active], n_sims, device,
                 noise_rng=rng, beta=beta)
        for i in active:
            t = trees[i]
            N = t.root.N
            tot = float(N.sum())
            if tot <= 0:  # degenerate; fall back to uniform over legal moves
                pi = t.root.legal.astype(np.float64)
                pi /= pi.sum()
            else:
                pi = N.astype(np.float64) / tot
            recs[i].append((t.root.state.encode().astype(np.int8),
                            pi.astype(np.float32), t.root.state.to_play))
            if move_i < temp_moves:
                a = int(rng.choice(N_ACT, p=pi / pi.sum()))
            else:
                a = int(N.argmax())
            logs[i].append(a)
            t.advance(a)
            if t.root.expanded:
                t.apply_noise(rng, DIR_ALPHA, DIR_EPS)
        active = [i for i in active if not trees[i].root.state.done]
        move_i += 1

    n_pos = sum(len(r) for r in recs)
    X = np.zeros((n_pos, 4, BOARD, BOARD), dtype=np.int8)
    PI = np.zeros((n_pos, N_ACT), dtype=np.float32)
    Z = np.zeros((n_pos,), dtype=np.float32)
    winners, lengths, j = [], [], 0
    for i in range(n_games):
        w = trees[i].root.state.winner
        winners.append(int(w))
        lengths.append(len(logs[i]))
        for x, pi, tp in recs[i]:
            X[j] = x
            PI[j] = pi
            Z[j] = 0.0 if w == 0 else (1.0 if w == tp else -1.0)
            j += 1
    return X, PI, Z, logs, winners, lengths


# --------------------------------------------------------------------------- #
# self-play worker processes
# --------------------------------------------------------------------------- #
def _worker(wid, gpu, task_q, res_q, ch, blocks):
    try:
        torch.cuda.set_device(gpu)
        dev = f"cuda:{gpu}"
        torch.set_num_threads(2)
        net = AZNet(ch, blocks).to(dev)
        net.eval()
        while True:
            task = task_q.get()
            if task is None:
                break
            sd, n_games, seed, n_sims, beta = task
            net.load_state_dict(sd)
            net.eval()
            rng = np.random.default_rng(seed)
            t0 = time.time()
            X, PI, Z, logs, winners, lengths = selfplay(
                net, n_games, n_sims, dev, rng, beta=beta)
            res_q.put((wid, X, PI, Z, logs, winners, lengths, time.time() - t0, None))
    except Exception as e:  # surface worker failures instead of hanging the parent
        import traceback
        res_q.put((wid, None, None, None, None, None, None, 0.0,
                   f"{e}\n{traceback.format_exc()}"))


class SelfPlayPool:
    """Persistent worker processes, one CUDA context each."""

    def __init__(self, gpus, ch, blocks):
        ctx = mp.get_context("spawn")
        self.task_q = ctx.Queue()
        self.res_q = ctx.Queue()
        self.procs = []
        for wid, gpu in enumerate(gpus):
            p = ctx.Process(target=_worker,
                            args=(wid, gpu, self.task_q, self.res_q, ch, blocks),
                            daemon=True)
            p.start()
            self.procs.append(p)
        self.n = len(self.procs)

    def run(self, net, n_games, n_sims, seed, beta, timeout=3600):
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        share = [n_games // self.n] * self.n
        for i in range(n_games % self.n):
            share[i] += 1
        for wid in range(self.n):
            self.task_q.put((sd, share[wid], seed * 1000 + wid, n_sims, beta))
        out = []
        for _ in range(self.n):
            r = self.res_q.get(timeout=timeout)
            if r[-1] is not None:
                raise RuntimeError(f"self-play worker {r[0]} failed: {r[-1]}")
            out.append(r)
        out.sort(key=lambda r: r[0])
        X = np.concatenate([r[1] for r in out])
        PI = np.concatenate([r[2] for r in out])
        Z = np.concatenate([r[3] for r in out])
        logs = [g for r in out for g in r[4]]
        winners = [w for r in out for w in r[5]]
        lengths = [l for r in out for l in r[6]]
        worker_s = [round(r[7], 1) for r in out]
        return X, PI, Z, logs, winners, lengths, worker_s

    def close(self):
        for _ in self.procs:
            self.task_q.put(None)
        for p in self.procs:
            p.join(timeout=30)


# --------------------------------------------------------------------------- #
# replay buffer with dihedral symmetry applied at sampling time
# --------------------------------------------------------------------------- #
class Buffer:
    def __init__(self, cap):
        self.x = np.zeros((cap, 4, BOARD, BOARD), dtype=np.int8)
        self.pi = np.zeros((cap, N_ACT), dtype=np.float32)
        self.z = np.zeros((cap,), dtype=np.float32)
        self.cap = cap
        self.n = 0
        self.ptr = 0

    def add_arrays(self, X, PI, Z):
        for s in range(0, len(X), self.cap):
            xs, ps, zs = X[s:s + self.cap], PI[s:s + self.cap], Z[s:s + self.cap]
            k = len(xs)
            end = self.ptr + k
            if end <= self.cap:
                self.x[self.ptr:end] = xs
                self.pi[self.ptr:end] = ps
                self.z[self.ptr:end] = zs
            else:
                a = self.cap - self.ptr
                self.x[self.ptr:] = xs[:a]
                self.pi[self.ptr:] = ps[:a]
                self.z[self.ptr:] = zs[:a]
                self.x[:k - a] = xs[a:]
                self.pi[:k - a] = ps[a:]
                self.z[:k - a] = zs[a:]
            self.ptr = end % self.cap
            self.n = min(self.n + k, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        xb = self.x[idx].astype(np.float32)
        pb = self.pi[idx]
        zb = self.z[idx]
        # 8 chunks, one dihedral transform each: cheap, vectorised, unbiased
        chunks = np.array_split(np.arange(batch), 8)
        xs, ps = [], []
        for g, sl in enumerate(chunks):
            k, flip = g % 4, g >= 4
            xc, pc = xb[sl], pb[sl].reshape(-1, BOARD, BOARD)
            if k:
                xc = np.rot90(xc, k, axes=(2, 3))
                pc = np.rot90(pc, k, axes=(1, 2))
            if flip:
                xc = xc[:, :, :, ::-1]
                pc = pc[:, :, ::-1]
            xs.append(np.ascontiguousarray(xc))
            ps.append(np.ascontiguousarray(pc).reshape(len(sl), -1))
        return np.concatenate(xs), np.concatenate(ps), zb


# --------------------------------------------------------------------------- #
# players (for evaluation)
# --------------------------------------------------------------------------- #
class RandomPlayer:
    name = "random"

    def move_batch(self, states, rng):
        return [int(rng.choice(np.flatnonzero(s.legal_mask()))) for s in states]


class RulePlayer:
    """Greedy play straight from the cold-start heuristic, no search."""

    name = "rule-greedy"

    def move_batch(self, states, rng):
        b = np.stack([s.board for s in states])
        tp = np.array([s.to_play for s in states], dtype=np.int8)
        return [int(p.argmax()) for p in rule_priors_batch(b, tp)]


class PureMCTSPlayer:
    """Classic MCTS: uniform priors, value from a uniform-random rollout."""

    def __init__(self, n_playout, c_puct=5.0):
        self.n_playout = n_playout
        self.c_puct = c_puct
        self.name = f"pure-mcts-{n_playout}"

    def _rollout(self, state, rng):
        s = state.clone()
        empty = np.flatnonzero(s.legal_mask())
        rng.shuffle(empty)
        for a in empty:
            s.play(int(a))
            if s.done:
                break
        return s.winner

    def move_batch(self, states, rng):
        return [self._move(s, rng) for s in states]

    def _move(self, state, rng):
        tree = Tree(state.clone(), self.c_puct)
        for _ in range(self.n_playout):
            path, leaf, tv = tree.select()
            if tv is not None:
                tree.backup(path, tv)
                continue
            legal = leaf.state.legal_mask()
            tree.expand(leaf, legal.astype(np.float64))
            w = self._rollout(leaf.state, rng)
            v = 0.0 if w == 0 else (1.0 if w == leaf.state.to_play else -1.0)
            tree.backup(path, v)
        return int(tree.root.N.argmax())


class AZPlayer:
    """Network + MCTS. beta is always 0 here: evaluation measures the model.

    temp=0 keeps the historical argmax behaviour: fully deterministic, so two
    AZPlayers replay the identical game for a given colour assignment and a
    round-robin only *orders* checkpoints. temp>0 samples moves from the visit
    distribution raised to 1/temp, which makes repeated games distinct and
    turns match scores into real measurements with error bars. Keep eval
    temperatures small (~0.3): the sampling is meant to diversify games, not
    to weaken play.
    """

    def __init__(self, net, n_sims, device, name="alphazero", temp=0.0):
        self.net = net
        self.n_sims = n_sims
        self.device = device
        self.name = name
        self.temp = temp

    def move_batch(self, states, rng):
        self.net.eval()
        trees = [Tree(s.clone()) for s in states]
        run_sims(self.net, trees, self.n_sims, self.device, noise_rng=None, beta=0.0)
        moves = []
        for t in trees:
            n = t.root.N.astype(np.float64)
            if self.temp > 0 and rng is not None and n.sum() > 0:
                w = (n / n.max()) ** (1.0 / self.temp)  # normalise before the
                w /= w.sum()                            # power to avoid overflow
                moves.append(int(rng.choice(N_ACT, p=w)))
            else:
                moves.append(int(n.argmax()))
        return moves

    def policy_and_value(self, state):
        """Raw net output plus MCTS visit distribution for one position."""
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(state.encode()[None]).to(self.device)
            logits, v = self.net(x)
            legal = state.legal_mask()
            p = torch.softmax(logits, dim=1).float().cpu().numpy()[0] * legal
            p = p / max(p.sum(), 1e-12)
            raw_v = float(v.float().cpu().numpy()[0])
        tree = Tree(state.clone())
        run_sims(self.net, [tree], self.n_sims, self.device, noise_rng=None, beta=0.0)
        n = tree.root.N
        visits = n / max(n.sum(), 1e-12)
        return p, visits, raw_v


def play_matches(pA, pB, n_games, rng, note=""):
    """pA and pB alternate colours. Returns (winsA, winsB, draws, games)."""
    states = [State() for _ in range(n_games)]
    a_is_black = [i % 2 == 0 for i in range(n_games)]
    logs = [[] for _ in range(n_games)]
    active = list(range(n_games))
    while active:
        a_idx = [i for i in active if (states[i].to_play == 1) == a_is_black[i]]
        b_idx = [i for i in active if i not in set(a_idx)]
        for player, idxs in ((pA, a_idx), (pB, b_idx)):
            if not idxs:
                continue
            mv = player.move_batch([states[i] for i in idxs], rng)
            for i, m in zip(idxs, mv):
                states[i].play(m)
                logs[i].append(int(m))
        active = [i for i in active if not states[i].done]

    wa = wb = dr = 0
    games = []
    for i in range(n_games):
        w = states[i].winner
        if w == 0:
            dr += 1
            res = "draw"
        elif (w == 1) == a_is_black[i]:
            wa += 1
            res = "A"
        else:
            wb += 1
            res = "B"
        games.append({"moves": logs[i], "a_is_black": a_is_black[i],
                      "result": res, "note": note})
    return wa, wb, dr, games


# --------------------------------------------------------------------------- #
# tactical probe positions: constructed patterns + exact checkers
#
# Tiers by depth of the forced line, so the report can show a difficulty
# ladder instead of a single 4/4 that saturates by iteration 5:
#   tier 1  win/lose this ply      (complete a five / block a five)
#   tier 2  forced win in <=3 plies (a move that leaves two winning cells:
#                                    open four or double four -- and blocking
#                                    the opponent from getting one)
#   tier 3  victory by continuous fours (VCF), >=2 of our stones deep
# Double-three / four-three combinations need a VCT search and are out of
# scope. The "good" sets are computed by the checkers below, not hand-listed,
# so the same generator works on any board size >= 9.
# --------------------------------------------------------------------------- #
def _mk(black, white, to_play=1):
    s = State()
    for (r, c) in black:
        s.board[r, c] = 1
    for (r, c) in white:
        s.board[r, c] = -1
    s.n_moves = len(black) + len(white)
    s.to_play = to_play
    return s


def _near_cells(board, radius=2):
    """Empty cells within Chebyshev `radius` of any stone. Every five-completing
    cell is within 1 of an own stone, every four-creating move within 2, so
    radius=2 is exhaustive for the checkers below."""
    cand = set()
    rs, cs = np.nonzero(board != 0)
    for r, c in zip(rs, cs):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD and 0 <= cc < BOARD and board[rr, cc] == 0:
                    cand.add(rr * BOARD + cc)
    return sorted(cand)


def _win_cells(board, player):
    """Empty cells where `player` completes a five immediately."""
    out = []
    for a in _near_cells(board, radius=1):
        r, c = divmod(a, BOARD)
        board[r, c] = player
        if _wins(board, r, c, player):
            out.append(a)
        board[r, c] = 0
    return out


def _double_threat_moves(board, player):
    """Moves after which `player` has >=2 winning cells (an open/double four)
    while the opponent has none: one block cannot cover both, and the opponent
    has no faster five of their own, so this forces a win in <=3 plies."""
    out = []
    for a in _near_cells(board):
        r, c = divmod(a, BOARD)
        board[r, c] = player
        if (not _wins(board, r, c, player)
                and len(_win_cells(board, player)) >= 2
                and not _win_cells(board, -player)):
            out.append(a)
        board[r, c] = 0
    return out


def _has_vcf(board, me, depth):
    """True if `me` (to move) has a victory by continuous fours within `depth`
    own stones. Every move considered must threaten a five while leaving the
    opponent without one (a four is only forcing if the opponent cannot win
    first), so the reply is forced and the search stays narrow."""
    if depth <= 0:
        return False
    opp = -me
    for a in _near_cells(board):
        r, c = divmod(a, BOARD)
        board[r, c] = me
        if _wins(board, r, c, me):
            board[r, c] = 0
            return True
        thr = _win_cells(board, me)
        ok = bool(thr) and not _win_cells(board, opp)
        if ok and len(thr) == 1:
            rb, cb = divmod(thr[0], BOARD)
            board[rb, cb] = opp
            ok = not _win_cells(board, opp) and _has_vcf(board, me, depth - 1)
            board[rb, cb] = 0
        board[r, c] = 0
        if ok:
            return True
    return False


def _vcf_starts(board, me, depth):
    """All first moves that begin a proven VCF within `depth` own stones."""
    good = []
    opp = -me
    for a in _near_cells(board):
        r, c = divmod(a, BOARD)
        board[r, c] = me
        if _wins(board, r, c, me):
            board[r, c] = 0
            good.append(a)
            continue
        thr = _win_cells(board, me)
        ok = bool(thr) and not _win_cells(board, opp)
        if ok and len(thr) == 1:
            rb, cb = divmod(thr[0], BOARD)
            board[rb, cb] = opp
            ok = not _win_cells(board, opp) and _has_vcf(board, me, depth - 1)
            board[rb, cb] = 0
        board[r, c] = 0
        if ok:
            good.append(a)
    return good


def _block_five_moves(board, me):
    """Moves that survive this ply: our own five, or the block that removes
    every five-completing cell of the opponent."""
    good = []
    for a in _near_cells(board):
        r, c = divmod(a, BOARD)
        board[r, c] = me
        if _wins(board, r, c, me) or not _win_cells(board, -me):
            good.append(a)
        board[r, c] = 0
    return good


def _defense_moves(board, me):
    """Moves after which the opponent has neither an immediate five nor any
    double-threat move -- i.e. we survive the next two plies."""
    good = []
    opp = -me
    for a in _near_cells(board):
        r, c = divmod(a, BOARD)
        board[r, c] = me
        if _wins(board, r, c, me) or (not _win_cells(board, opp)
                                      and not _double_threat_moves(board, opp)):
            good.append(a)
        board[r, c] = 0
    return good


def tactical_positions():
    """Constructed probes on the current BOARD, one list, tier-tagged.

    Patterns are written in the coordinates of a 9x9 window and shifted to the
    board centre, each in two orientations (identity and transpose). Noise
    stones keep the colour counts equal (black to play) and sit far enough
    from the action to stay out of every checker's radius."""
    assert BOARD >= 9, "tactical probes need at least a 9x9 board"
    d = (BOARD - 9) // 2

    def sh(cells):
        return [(r + d, c + d) for r, c in cells]

    def tr(cells):
        return [(c, r) for r, c in cells]

    base = [
        # tier 1: four with one blocked end -> complete the five
        {"id": "t1_win", "tier": 1, "kind": "attack",
         "desc": "黑四连、一端被封 → 落子成五",
         "black": [(4, 2), (4, 3), (4, 4), (4, 5)], "white": [(4, 1)],
         "check": lambda b: _win_cells(b, 1)},
        # tier 1: opponent four -> the block is forced
        {"id": "t1_block", "tier": 1, "kind": "defend",
         "desc": "白四连、一端已被封 → 必须堵",
         "black": [(2, 1)], "white": [(2, 2), (2, 3), (2, 4), (2, 5)],
         "check": lambda b: _block_five_moves(b, 1)},
        # tier 2: open three -> make an open four (two winning cells)
        {"id": "t2_open4", "tier": 2, "kind": "attack",
         "desc": "黑活三 → 延伸成活四（两个成五点）",
         "black": [(4, 3), (4, 4), (4, 5)], "white": [],
         "check": lambda b: _double_threat_moves(b, 1)},
        # tier 2: opponent open three -> deny every double-threat continuation
        {"id": "t2_block3", "tier": 2, "kind": "defend",
         "desc": "白活三 → 堵住，不给活四",
         "black": [], "white": [(3, 3), (3, 4), (3, 5)],
         "check": lambda b: _defense_moves(b, 1)},
        # tier 3: no single move wins, but a four forces the block and the
        # follow-up four is double -- VCF two stones deep
        {"id": "t3_vcf2", "tier": 3, "kind": "attack",
         "desc": "连续冲四取胜（VCF 深度 2）：先冲四逼堵，再双四",
         "black": [(4, 2), (4, 3), (4, 4), (5, 5), (6, 5)],
         "white": [(4, 1)],
         "check": lambda b: _vcf_starts(b, 1, 3)},
    ]

    # filler stones balance the colour counts (equal counts -> black to play).
    # Pattern stones in both orientations live in rows/cols 1-6, so cells on
    # rows {0,7,8} x cols {0,7,8} (plus (8,4), 4+ away from anything) can
    # never extend a tactical line. Same-colour fillers sit 7+ apart.
    fill_b = [(0, 0), (8, 8), (7, 0)]
    fill_w = [(0, 8), (8, 0), (7, 8), (8, 4)]

    pos = []
    for spec in base:
        for flip, xf in (("", lambda x: x), ("_tr", tr)):
            black, white = list(xf(spec["black"])), list(xf(spec["white"]))
            need = len(black) - len(white)
            assert -len(fill_b) <= need <= len(fill_w), spec["id"]
            if need > 0:
                white += fill_w[:need]
            elif need < 0:
                black += fill_b[:-need]
            black, white = sh(black), sh(white)
            cells = black + white
            assert len(set(cells)) == len(cells), f"{spec['id']}: overlap"
            s = _mk(black, white)
            good = spec["check"](s.board)
            pos.append({"id": spec["id"] + flip, "tier": spec["tier"],
                        "kind": spec["kind"], "desc": spec["desc"],
                        "state": s, "good": [int(a) for a in good]})

    # sanity: every probe solvable, every good move legal, tiers honest
    for p in pos:
        s = p["state"]
        assert p["good"], f"{p['id']}: no correct move found by checker"
        assert not s.done and s.to_play == 1
        assert (s.board != 0).sum() == s.n_moves
        for a in p["good"]:
            assert s.legal_mask()[a], p["id"]
        if p["tier"] == 1 and p["kind"] == "attack":
            t = s.clone()
            t.play(p["good"][0])
            assert t.done and t.winner == 1, p["id"]
        if p["tier"] == 3:
            assert not _win_cells(s.board, 1), p["id"]
            assert not _double_threat_moves(s.board, 1), \
                f"{p['id']}: a single move already wins, that is tier 2"
        if p["kind"] == "defend":
            # the threat must be real: most legal moves lose
            assert len(p["good"]) <= len(_near_cells(s.board)) // 2, p["id"]
    return pos


# --------------------------------------------------------------------------- #
# Elo from a round-robin (Bradley-Terry MLE, first entry anchored at 0)
# --------------------------------------------------------------------------- #
def fit_elo(n, results, iters=4000, lr=1.0):
    r = np.zeros(n)
    for _ in range(iters):
        g = np.zeros(n)
        for i, j, wi, wj, dr in results:
            si = wi + 0.5 * dr
            tot = wi + wj + dr
            if tot == 0:
                continue
            pe = 1.0 / (1.0 + 10 ** (-(r[i] - r[j]) / 400.0))
            d = (si - tot * pe)
            g[i] += d
            g[j] -= d
        r += lr * g
        r -= r[0]
    return [round(float(v), 1) for v in r]


def beta_at(it):
    if BETA0 <= 0 or RULE_ITERS <= 0:
        return 0.0
    return round(BETA0 * max(0.0, 1.0 - (it - 1) / RULE_ITERS), 4)


# --------------------------------------------------------------------------- #
# self-test: rules, encoding symmetry, rule oracle
# --------------------------------------------------------------------------- #
def selftest():
    for name, cells in (
        ("row", [(4, c) for c in range(5)]),
        ("col", [(r, 4) for r in range(5)]),
        ("diag", [(i, i) for i in range(5)]),
        ("anti", [(i, 8 - i) for i in range(5)]),
    ):
        s = State()
        filler = [(7, c) for c in range(5)]
        for k in range(5):
            if k > 0:
                assert not s.done, name
            s.play(cells[k][0] * BOARD + cells[k][1])
            if k < 4:
                s.play(filler[k][0] * BOARD + filler[k][1])
        assert s.done and s.winner == 1, f"{name} win not detected"
    s = State()
    for k in range(4):
        s.play(4 * BOARD + k)
        s.play(0 * BOARD + k)
    assert not s.done, "four in a row wrongly counted as a win"

    # encode() must commute with rot90: (r,c) -> (BOARD-1-c, r)
    s1, s2 = State(), State()
    for m in [40, 30, 41, 31, 50, 22]:
        r, c = divmod(m, BOARD)
        s1.play(m)
        s2.play((BOARD - 1 - c) * BOARD + r)
    assert np.allclose(s2.encode(), np.rot90(s1.encode(), 1, axes=(1, 2))), \
        "encode/rot90 mismatch"

    # rule oracle must find the win, the block, and stay legal
    probes = tactical_positions()
    b = np.stack([p["state"].board for p in probes])
    tp = np.array([p["state"].to_play for p in probes], dtype=np.int8)
    rp = rule_priors_batch(b, tp)
    for p, row in zip(probes, rp):
        assert abs(row.sum() - 1.0) < 1e-5, "rule prior not normalised"
        occupied = (p["state"].board.reshape(-1) != 0)
        assert row[occupied].sum() < 1e-9, "rule prior on occupied cell"
        # the greedy heuristic sees one ply: it must solve tier 1, is expected
        # to solve tier 2, and has no VCF search for tier 3
        if p["tier"] == 1:
            assert int(row.argmax()) in p["good"], f"rule oracle missed {p['id']}"
        elif int(row.argmax()) not in p["good"]:
            print(f"[selftest] note: rule oracle does not solve {p['id']} "
                  f"(tier {p['tier']}, fine)", flush=True)
    # rule prior on an empty board must be legal and centre-weighted
    e = State()
    rp0 = rule_priors_batch(e.board[None], np.array([1], dtype=np.int8))[0]
    assert abs(rp0.sum() - 1.0) < 1e-5
    assert int(rp0.argmax()) == (BOARD // 2) * BOARD + BOARD // 2, "not centre"
    print("[selftest] rules + encoding + rule oracle OK", flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    selftest()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")

    device = DEVICE if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    os.makedirs("results", exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    net = AZNet().to(device)
    n_par = sum(p.numel() for p in net.parameters())
    if RESUME_ITER > 0:
        rp = f"{CKPT_DIR}/iter{RESUME_ITER:03d}.pt"
        net.load_state_dict(torch.load(rp, map_location=device))
        print(f"[resume] loaded {rp}; replay buffer starts empty, optimizer "
              f"moments reset", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(ITERS * 0.5), int(ITERS * 0.75)], gamma=0.5)
    for _ in range(RESUME_ITER):   # replay the LR schedule up to the resume point
        sched.step()
    buf = Buffer(BUFFER_CAP)
    pool = SelfPlayPool(GPUS, CHANNELS, BLOCKS)

    M = {
        "algo": "AlphaZero", "tag": TAG,
        "game": f"Gomoku {BOARD}x{BOARD} / {N_IN_ROW}-in-a-row",
        "device": device, "gpus": GPUS, "workers": pool.n,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "net": {"channels": CHANNELS, "blocks": BLOCKS, "params": n_par,
                "norm": "GroupNorm", "planes": 4},
        "cfg": {"iters": ITERS, "games_per_iter": GAMES_PER_ITER, "sims": N_SIMS,
                "c_puct": C_PUCT, "dirichlet": [DIR_ALPHA, DIR_EPS],
                "temp_moves": TEMP_MOVES, "batch": BATCH,
                "train_steps": TRAIN_STEPS, "lr": LR, "buffer": BUFFER_CAP,
                "eval_sims": EVAL_SIMS, "board": BOARD, "n_in_row": N_IN_ROW,
                "beta0": BETA0, "rule_iters": RULE_ITERS, "seed": SEED},
        "phase_start": time.time(),
        "iterations": [], "evals": [], "tactical": [], "sample_games": [],
        "phases": [], "elo": [], "anchors": [],
    }
    print(f"[init] {M['game']} | net {n_par/1e6:.2f}M | {pool.n} workers on GPUs {GPUS} "
          f"| beta0={BETA0} rule_iters={RULE_ITERS} | tag={TAG}", flush=True)

    probes = tactical_positions()
    M["probe_meta"] = [{"id": p["id"], "desc": p["desc"], "good": p["good"],
                        "tier": p["tier"], "kind": p["kind"],
                        "board": [int(v) for v in p["state"].board.reshape(-1)]}
                       for p in probes]
    snapshots = []

    def snapshot(tag, it):
        path = f"{CKPT_DIR}/{tag}.pt"
        torch.save(net.state_dict(), path)
        snapshots.append({"tag": tag, "iter": it, "path": path})

    if RESUME_ITER > 0:
        # carry over the recorded history and the snapshot list so anchors and
        # the final round-robin see the pre-resume checkpoints
        if os.path.exists(OUT_JSON):
            with open(OUT_JSON) as f:
                old = json.load(f)
            for k in ("iterations", "evals", "tactical", "sample_games",
                      "phases", "anchors"):
                M[k] = [r for r in old.get(k, [])
                        if r.get("iter", 0) <= RESUME_ITER]
            M["phase_start"] = old.get("phase_start", M["phase_start"])
            M["resumed_at"] = old.get("resumed_at", []) + [
                {"iter": RESUME_ITER, "time": time.time(),
                 "workers": len(GPUS), "gpus": GPUS}]
            print(f"[resume] carried {len(M['iterations'])} iterations / "
                  f"{len(M['evals'])} evals from {OUT_JSON}", flush=True)
        for k in range(0, RESUME_ITER + 1):
            p = f"{CKPT_DIR}/iter{k:03d}.pt"
            if os.path.exists(p):
                snapshots.append({"tag": f"iter{k:03d}", "iter": k, "path": p})

    def dump():
        M["phase_end"] = time.time()
        tmp = OUT_JSON + ".tmp"
        with open(tmp, "w") as f:
            json.dump(M, f)
        os.replace(tmp, OUT_JSON)

    def probe(it):
        pl = AZPlayer(net, EVAL_SIMS, device)
        rec = {"iter": it, "positions": []}
        n_raw = n_mcts = 0
        for p in probes:
            pol, vis, val = pl.policy_and_value(p["state"])
            raw_a, mcts_a = int(pol.argmax()), int(vis.argmax())
            ok_raw, ok_mcts = raw_a in p["good"], mcts_a in p["good"]
            n_raw += ok_raw
            n_mcts += ok_mcts
            rec["positions"].append({
                "id": p["id"], "raw_move": raw_a, "mcts_move": mcts_a,
                "raw_ok": bool(ok_raw), "mcts_ok": bool(ok_mcts),
                "value": round(val, 4),
                "policy": [round(float(v), 5) for v in pol],
                "visits": [round(float(v), 5) for v in vis],
            })
        rec["raw_acc"] = n_raw / len(probes)
        rec["mcts_acc"] = n_mcts / len(probes)
        tiers = {}
        for p, q in zip(probes, rec["positions"]):
            t = tiers.setdefault(p["tier"], [0, 0, 0])
            t[0] += 1
            t[1] += q["raw_ok"]
            t[2] += q["mcts_ok"]
        rec["tier_acc"] = {str(k): {"n": n, "raw": round(r / n, 4),
                                    "mcts": round(m / n, 4)}
                           for k, (n, r, m) in sorted(tiers.items())}
        M["tactical"].append(rec)
        by_tier = " ".join(f"T{k}:{v['raw']:.0%}/{v['mcts']:.0%}"
                           for k, v in rec["tier_acc"].items())
        print(f"  [probe] raw {n_raw}/{len(probes)}  mcts {n_mcts}/{len(probes)}"
              f"  ({by_tier})", flush=True)

    if RESUME_ITER == 0:
        snapshot("iter000", 0)
        probe(0)

    try:
        for it in range(RESUME_ITER + 1, ITERS + 1):
            beta = beta_at(it)

            t0 = time.time()
            X, PI, Z, logs, winners, lengths, worker_s = pool.run(
                net, GAMES_PER_ITER, N_SIMS, SEED + it, beta)
            t1 = time.time()
            M["phases"].append({"kind": "selfplay", "iter": it, "t0": t0, "t1": t1})
            buf.add_arrays(X, PI, Z)
            black_w = sum(1 for w in winners if w == 1)
            white_w = sum(1 for w in winners if w == -1)
            draws = sum(1 for w in winners if w == 0)

            net.train()
            t2 = time.time()
            pl_sum = vl_sum = ent_sum = 0.0
            zs_all, vs_all = [], []
            for _ in range(TRAIN_STEPS):
                xb, pb, zb = buf.sample(BATCH, rng)
                x = torch.from_numpy(xb).to(device)
                pt = torch.from_numpy(pb).to(device)
                zt = torch.from_numpy(zb).to(device)
                logits, v = net(x)
                logp = F.log_softmax(logits, dim=1)
                p_loss = -(pt * logp).sum(dim=1).mean()
                v_loss = F.mse_loss(v, zt)
                loss = p_loss + v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                with torch.no_grad():
                    ent = -(logp.exp() * logp).sum(dim=1).mean()
                pl_sum += p_loss.detach().item()
                vl_sum += v_loss.detach().item()
                ent_sum += ent.item()
                zs_all.append(zb)
                vs_all.append(v.detach().float().cpu().numpy())
            sched.step()
            t3 = time.time()
            M["phases"].append({"kind": "train", "iter": it, "t0": t2, "t1": t3})

            z_cat, v_cat = np.concatenate(zs_all), np.concatenate(vs_all)
            ev = 1.0 - float(np.var(z_cat - v_cat) / max(np.var(z_cat), 1e-9))
            rec = {
                "iter": it, "beta": beta,
                "policy_loss": round(pl_sum / TRAIN_STEPS, 5),
                "value_loss": round(vl_sum / TRAIN_STEPS, 5),
                "total_loss": round((pl_sum + vl_sum) / TRAIN_STEPS, 5),
                "entropy": round(ent_sum / TRAIN_STEPS, 5),
                "explained_var": round(ev, 4),
                "lr": opt.param_groups[0]["lr"],
                "games": GAMES_PER_ITER, "positions": int(len(X)),
                "cum_games": it * GAMES_PER_ITER, "buffer": buf.n,
                "avg_len": round(float(np.mean(lengths)), 2),
                "min_len": int(np.min(lengths)), "max_len": int(np.max(lengths)),
                "black_wins": black_w, "white_wins": white_w, "draws": draws,
                "selfplay_s": round(t1 - t0, 2), "train_s": round(t3 - t2, 2),
                "worker_s": worker_s,
            }
            M["iterations"].append(rec)
            print(f"[{TAG} iter {it}/{ITERS}] b={beta:.3f} loss {rec['total_loss']:.4f} "
                  f"(p {rec['policy_loss']:.4f} v {rec['value_loss']:.4f}) "
                  f"ent {rec['entropy']:.3f} ev {ev:+.3f} | len {rec['avg_len']:.1f} "
                  f"B/W/D {black_w}/{white_w}/{draws} | sp {rec['selfplay_s']:.0f}s "
                  f"tr {rec['train_s']:.0f}s", flush=True)

            for gi in (0, 1):
                M["sample_games"].append({
                    "iter": it, "kind": "selfplay",
                    "moves": [int(a) for a in logs[gi]], "winner": winners[gi]})

            if it % EVAL_EVERY == 0 or it == ITERS:
                t4 = time.time()
                az = AZPlayer(net, EVAL_SIMS, device)
                ev_rec = {"iter": it, "cum_games": it * GAMES_PER_ITER,
                          "elapsed_s": round(time.time() - M["phase_start"], 1),
                          "matches": []}
                opps = [(RandomPlayer(), EVAL_GAMES_RAND),
                        (RulePlayer(), EVAL_GAMES_RAND),
                        (PureMCTSPlayer(PURE_PLAYOUTS), EVAL_GAMES_MCTS)]
                for opp, ngames in opps:
                    wa, wb, dr, games = play_matches(az, opp, ngames, rng, opp.name)
                    ev_rec["matches"].append({
                        "opponent": opp.name, "games": ngames,
                        "win": wa, "loss": wb, "draw": dr,
                        "win_rate": round((wa + 0.5 * dr) / ngames, 4)})
                    print(f"  [eval] vs {opp.name}: {wa}W-{wb}L-{dr}D", flush=True)
                    won = [g for g in games if g["result"] == "A"]
                    if won and opp.name.startswith("pure"):
                        M["sample_games"].append({
                            "iter": it, "kind": f"vs-{opp.name}",
                            "moves": won[0]["moves"],
                            "az_is_black": bool(won[0]["a_is_black"]),
                            "winner": 1 if won[0]["a_is_black"] else -1})
                t5 = time.time()
                M["phases"].append({"kind": "eval", "iter": it, "t0": t4, "t1": t5})
                ev_rec["seconds"] = round(t5 - t4, 1)
                M["evals"].append(ev_rec)

                # rolling anchor ladder: play the last ANCHOR_K checkpoints with
                # a small temperature. score > 0.5 against the newest anchor
                # means the net is still improving -- unlike the fixed
                # baselines, this measurement never saturates.
                if ANCHOR_GAMES > 0:
                    anchors = [s for s in snapshots if s["iter"] < it][-ANCHOR_K:]
                    if anchors:
                        cur = AZPlayer(net, ANCHOR_SIMS, device,
                                       f"iter{it:03d}", temp=ANCHOR_TEMP)
                        ta = time.time()
                        for s in anchors:
                            m = AZNet().to(device)
                            m.load_state_dict(
                                torch.load(s["path"], map_location=device))
                            m.eval()
                            opp = AZPlayer(m, ANCHOR_SIMS, device, s["tag"],
                                           temp=ANCHOR_TEMP)
                            wa, wb, dr, _ = play_matches(
                                cur, opp, ANCHOR_GAMES, rng, "anchor")
                            M["anchors"].append({
                                "iter": it, "vs_iter": s["iter"],
                                "games": ANCHOR_GAMES, "win": wa, "loss": wb,
                                "draw": dr,
                                "score": round((wa + 0.5 * dr) / ANCHOR_GAMES, 4),
                                "sims": ANCHOR_SIMS, "temp": ANCHOR_TEMP})
                            print(f"  [anchor] vs {s['tag']}: {wa}W-{wb}L-{dr}D",
                                  flush=True)
                        M["phases"].append({"kind": "eval", "iter": it,
                                            "t0": ta, "t1": time.time()})

                probe(it)
                snapshot(f"iter{it:03d}", it)
            dump()

        print("[final] harder baseline", flush=True)
        az = AZPlayer(net, EVAL_SIMS, device)
        t0 = time.time()
        wa, wb, dr, games = play_matches(
            az, PureMCTSPlayer(PURE_PLAYOUTS_HARD), EVAL_GAMES_MCTS, rng, "hard")
        M["final_hard"] = {"opponent": f"pure-mcts-{PURE_PLAYOUTS_HARD}",
                           "games": EVAL_GAMES_MCTS, "win": wa, "loss": wb,
                           "draw": dr,
                           "win_rate": round((wa + 0.5 * dr) / EVAL_GAMES_MCTS, 4),
                           "seconds": round(time.time() - t0, 1)}
        print(f"  vs pure-mcts-{PURE_PLAYOUTS_HARD}: {wa}W-{wb}L-{dr}D", flush=True)
        won = [g for g in games if g["result"] == "A"]
        if won:
            M["sample_games"].append({
                "iter": ITERS, "kind": f"vs-pure-mcts-{PURE_PLAYOUTS_HARD}",
                "moves": won[0]["moves"],
                "az_is_black": bool(won[0]["a_is_black"]),
                "winner": 1 if won[0]["a_is_black"] else -1})
        dump()

        print("[final] Elo round-robin", flush=True)
        t0 = time.time()
        picks = snapshots if len(snapshots) <= 6 else \
            [snapshots[round(i * (len(snapshots) - 1) / 5)] for i in range(6)]
        seen, uniq = set(), []
        for s in picks:
            if s["tag"] not in seen:
                seen.add(s["tag"])
                uniq.append(s)
        nets = []
        for s in uniq:
            m = AZNet().to(device)
            m.load_state_dict(torch.load(s["path"], map_location=device))
            m.eval()
            nets.append(AZPlayer(m, ELO_SIMS, device, s["tag"], temp=ELO_TEMP))
        rr = []
        for i in range(len(nets)):
            for j in range(i + 1, len(nets)):
                wa, wb, dr, _ = play_matches(nets[i], nets[j], ELO_GAMES, rng)
                rr.append((i, j, wa, wb, dr))
                print(f"  {uniq[i]['tag']} vs {uniq[j]['tag']}: {wa}-{wb}-{dr}",
                      flush=True)
        elo = fit_elo(len(nets), rr)
        M["elo"] = [{"tag": uniq[k]["tag"], "iter": uniq[k]["iter"], "elo": elo[k]}
                    for k in range(len(nets))]
        M["elo_detail"] = {"sims": ELO_SIMS, "games_per_pair": ELO_GAMES,
                           "temp": ELO_TEMP,
                           "pairs": [{"i": i, "j": j, "wi": wi, "wj": wj, "d": d}
                                     for i, j, wi, wj, d in rr],
                           "seconds": round(time.time() - t0, 1)}
        print("  elo:", [(e["tag"], e["elo"]) for e in M["elo"]], flush=True)
        dump()
    finally:
        pool.close()
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

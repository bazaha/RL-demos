"""Deep-model GPU validation: ResNet-18 on CIFAR-10, single H20, AMP bf16.

Writes results/dl_metrics.json with per-step loss, per-epoch accuracy/throughput.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.transforms import v2

EPOCHS = 5
BATCH = 512
LR = 0.2
DATA = "data/cifar10.npz"
OUT = "results/dl_metrics.json"

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Cifar(Dataset):
    def __init__(self, x, y, train):
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        self.y = torch.from_numpy(y)
        aug = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()] if train else []
        self.tf = v2.Compose(aug + [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


def make_model():
    m = resnet18(num_classes=10)
    # CIFAR stem: 3x3 conv, no maxpool (32x32 inputs)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


def main():
    assert torch.cuda.is_available()
    device = "cuda:0"
    torch.backends.cudnn.benchmark = True

    d = np.load(DATA)
    train_ds = Cifar(d["train_x"], d["train_y"], train=True)
    test_ds = Cifar(d["test_x"], d["test_y"], train=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=8,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=1024, num_workers=4, pin_memory=True)

    model = make_model().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps_per_epoch = len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per_epoch)

    metrics = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "model": "ResNet-18 (CIFAR stem)",
        "dataset": "CIFAR-10",
        "phase_start": time.time(),
        "steps": [],       # {step, loss, lr}
        "epoch_stats": [], # {epoch, train_loss, test_acc, images_per_sec, epoch_seconds}
    }

    step = 0
    for epoch in range(EPOCHS):
        t0 = time.time()
        seen = 0
        losses = []
        for x, y in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            seen += y.numel()
            losses.append(loss.item())
            if step % 10 == 0:
                metrics["steps"].append({"step": step, "loss": round(loss.item(), 4),
                                         "lr": round(sched.get_last_lr()[0], 5)})
        torch.cuda.synchronize()
        dt = time.time() - t0
        acc = evaluate(model, test_dl, device)
        stat = {"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 4),
                "test_acc": round(acc, 4), "images_per_sec": round(seen / dt), "epoch_seconds": round(dt, 1)}
        metrics["epoch_stats"].append(stat)
        print(f"epoch {epoch+1}/{EPOCHS} loss={stat['train_loss']:.4f} acc={acc:.4f} "
              f"{stat['images_per_sec']} img/s ({dt:.1f}s)", flush=True)

    metrics["phase_end"] = time.time()
    metrics["final_acc"] = metrics["epoch_stats"][-1]["test_acc"]
    metrics["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(metrics, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

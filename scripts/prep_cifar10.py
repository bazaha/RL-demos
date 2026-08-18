"""Convert HF-mirror CIFAR-10 parquet files into a single npz for fast loading."""
import io
import sys

import numpy as np
import pandas as pd
from PIL import Image


def load_split(path):
    df = pd.read_parquet(path)
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in df["img"]])
    labels = df["label"].to_numpy(dtype=np.int64)
    return imgs, labels


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    train_x, train_y = load_split(f"{data_dir}/cifar10-train.parquet")
    test_x, test_y = load_split(f"{data_dir}/cifar10-test.parquet")
    print(f"train {train_x.shape} test {test_x.shape}")
    np.savez(f"{data_dir}/cifar10.npz", train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    print("saved", f"{data_dir}/cifar10.npz")


if __name__ == "__main__":
    main()

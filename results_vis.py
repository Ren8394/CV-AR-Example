from pathlib import Path
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

CLASS_LIST = [
    "sitting", "using_laptop", "hugging", "sleeping", "drinking", 
    "clapping", "dancing", "cycling", "calling", "laughing",
    "eating", "fighting", "listening_to_music", "running", "texting"
]

if __name__ == "__main__":
    model_name = "vgg"

    results_dir = Path("results")
    t = np.loadtxt(results_dir.joinpath(f"true_class_count_{model_name}.txt"), dtype=int).tolist()
    p = np.loadtxt(results_dir.joinpath(f"pred_class_count_{model_name}.txt"), dtype=int).tolist()

    print(classification_report(t, p, zero_division=0, target_names=CLASS_LIST))

    tl = np.loadtxt(results_dir.joinpath(f"train_loss_{model_name}.txt"), dtype=float).tolist()
    vl = np.loadtxt(results_dir.joinpath(f"val_loss_{model_name}.txt"), dtype=float).tolist()

    plt.figure(figsize=(10, 5))
    plt.plot(tl, label="train loss")
    plt.plot(vl, label="val loss")
    plt.xlabel("epoch")
    plt.xticks(np.arange(0, len(tl), 5))
    plt.ylabel("loss")
    plt.legend()
    plt.show()
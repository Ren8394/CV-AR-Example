from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import pandas as pd

CLASS_LIST = [
    "sitting", "using_laptop", "hugging", "sleeping", "drinking", 
    "clapping", "dancing", "cycling", "calling", "laughing",
    "eating", "fighting", "listening_to_music", "running", "texting"
]

if __name__ == "__main__":
    folder_dir = Path.cwd().joinpath("data/Human Action Recognition")

    df = pd.read_csv(folder_dir.joinpath("Training_set.csv"), header=0)

    # split train, val, and test
    # preserve class distribution
    grouped = df.groupby("label")
    group_sizes = grouped.size()

    # set train, val, and test size
    sizes = [int(.8 * df.shape[0]), int(.1 * df.shape[0]), int(.1 * df.shape[0])]

    small_dfs = []
    for size in sizes:
        sample_sizes = (group_sizes * size / len(df)).round().astype(int)

        sampled = [g.sample(n=sample_sizes[n], replace=False, random_state=2024) for n, g in grouped]
        small_df = pd.concat(sampled).reset_index(drop=True)
        small_dfs.append(small_df)
        
        for i, k in enumerate(grouped.groups.keys()):
            grouped.get_group(k).drop(sampled[i].index, inplace=True)
            group_sizes[k] -= sample_sizes[k]

    train_df, val_df, test_df = small_dfs
    train_df.to_csv(folder_dir.joinpath("train.csv"), index=False, header=True)
    val_df.to_csv(folder_dir.joinpath("val.csv"), index=False, header=True)
    test_df.to_csv(folder_dir.joinpath("test.csv"), index=False, header=True)

    print(train_df.groupby("label").size())
    print(val_df.groupby("label").size())
    print(test_df.groupby("label").size())
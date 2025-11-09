import pandas as pd
import torch
from torch.utils.data import Dataset


class HorseRacingDataset(Dataset):
    """カスタムデータセット"""

    def __init__(self, odds, rankings):
        self.odds = odds
        self.rankings = rankings

    def __len__(self):
        return len(self.odds)

    def __getitem__(self, idx):
        return self.odds[idx], self.rankings[idx]


def get_datasets():
    """データセット準備"""
    frame = pd.read_csv("data/races.csv")
    # "place", "odds"列が数値でない行をドロップ
    frame = frame.loc[pd.to_numeric(frame["place"], errors="coerce").notna()]
    frame = frame.loc[pd.to_numeric(frame["odds"], errors="coerce").notna()]
    frame = frame.assign(place=frame["place"].astype(int), odds=frame["odds"].astype(float))

    grouped = frame.groupby("id")
    odds_list = []
    rank_list = []
    for _, group in grouped:
        odds = list(group["odds"])
        # 1-basedの着順を0-basedのインデックスに変換
        ranks = list(group["place"] - 1)
        odds_list.append(odds)
        rank_list.append(ranks)

    # Tensorに変換（18頭にパディング）
    n = len(odds_list)
    odds_tensor = torch.zeros(n, 18)  # [n, 18]
    rank_tensor_one_hot = torch.zeros(n, 18, 18)  # [n, 18, 18]

    for i, (odds, ranks) in enumerate(zip(odds_list, rank_list)):
        num_horses = len(odds)
        odds_tensor[i, :num_horses] = torch.tensor(odds)
        # horse_numberをインデックスとして使用
        horse_indices = list(range(num_horses))
        permuted_ranks = [0] * num_horses
        for horse_idx, rank in zip(horse_indices, ranks):
            if rank < num_horses:
                permuted_ranks[rank] = horse_idx

        # one-hotエンコーディング
        for horse_idx, rank in enumerate(ranks):
            if rank < 18:
                rank_tensor_one_hot[i, horse_idx, rank] = 1.0

    # 訓練・検証分割（8:2）
    split = int(n * 0.8)

    train_dataset = HorseRacingDataset(odds_tensor[:split], rank_tensor_one_hot[:split])
    test_dataset = HorseRacingDataset(odds_tensor[split:], rank_tensor_one_hot[split:])

    return train_dataset, test_dataset

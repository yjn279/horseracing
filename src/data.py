import torch
from torch.utils.data import Dataset

# サンプルデータ（3パターン）
SAMPLES = [
    # パターン1: 本命（10頭）
    ([3.5, 5.2, 8.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    
    # パターン2: 混戦（10頭）
    ([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
     [2, 0, 1, 3, 4, 5, 6, 7, 8, 9]),
    
    # パターン3: 穴（10頭）
    ([4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 3.0, 20.0, 25.0],
     [7, 0, 1, 2, 3, 4, 5, 6, 8, 9]),
]

class HorseRacingDataset(Dataset):
    """カスタムデータセット"""
    def __init__(self, odds, rankings):
        self.odds = odds
        self.rankings = rankings

    def __len__(self):
        return len(self.odds)

    def __getitem__(self, idx):
        return self.odds[idx], self.rankings[idx]


def get_datasets(n_repeat=300):
    """データセット準備"""
    odds_list = []
    rank_list = []
    
    # 各パターンをn_repeat回繰り返し
    for odds, ranks in SAMPLES:
        for _ in range(n_repeat):
            odds_list.append(odds)
            rank_list.append(ranks)
    
    # Tensorに変換（18頭にパディング）
    n = len(odds_list)  # n = 900
    odds_tensor = torch.zeros(n, 18)  # [900, 18]
    rank_tensor_one_hot = torch.zeros(n, 18, 18)  # [900, 18, 18]
    
    for i, (odds, ranks) in enumerate(zip(odds_list, rank_list)):
        odds_tensor[i, :len(odds)] = torch.tensor(odds)  # [10] → [18] (padded)
        for horse_idx, rank in enumerate(ranks):
            rank_tensor_one_hot[i, horse_idx, rank] = 1.0  # one-hot
    
    # 訓練・検証分割（8:2）
    split = int(n * 0.8)  # 720
    
    train_dataset = HorseRacingDataset(odds_tensor[:split], rank_tensor_one_hot[:split])
    test_dataset = HorseRacingDataset(odds_tensor[split:], rank_tensor_one_hot[split:])
    
    return train_dataset, test_dataset

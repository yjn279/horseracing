import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


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


# TODO: オッズの逆数をinputにする
def plackett_luce_model(odds):
    """単勝オッズからPlackett-Luceモデルで1-3着の確率行列を生成"""
    n_horses = len(odds)
    strengths = 1 / torch.tensor(odds, dtype=torch.float32)
    
    # [馬, 着順] の確率行列
    prob_matrix = torch.zeros((n_horses, 3))
    
    # 1着の確率
    total_strength = torch.sum(strengths)
    prob_matrix[:, 0] = strengths / total_strength
    
    # 2着、3着の確率
    for rank in range(1, 3):
        # 各馬iがrank着になる確率を計算
        prob_rank = torch.zeros(n_horses)
        for i in range(n_horses):
            # i以外の馬kがrank-1着になる確率
            prob_k_prev_rank = prob_matrix[:, rank - 1]
            
            # kがrank-1着になったという条件のもとで、iがrank着になる確率
            # (k以外の馬の強さ合計からiの強さを割る)
            cond_prob_i_given_k = strengths[i] / (total_strength - strengths)
            
            # iとkが同じ場合は確率0
            cond_prob_i_given_k[i] = 0
            
            # P(iがrank着) = Σk [ P(iがrank着 | kがrank-1着) * P(kがrank-1着) ]
            prob_rank[i] = torch.sum(cond_prob_i_given_k * prob_k_prev_rank)
            
        prob_matrix[:, rank] = prob_rank

    # 正規化
    for rank in range(3):
        col_sum = torch.sum(prob_matrix[:, rank])
        if col_sum > 0:
            prob_matrix[:, rank] /= col_sum
            
    return prob_matrix

def get_datasets(n_repeat=300):
    """データセット準備"""
    odds_list = []
    rank_list = []

    for odds, ranks in SAMPLES:
        for _ in range(n_repeat):
            odds_list.append(odds)
            rank_list.append(ranks)
    
    # Tensorに変換（18頭にパディング）
    n = len(odds_list)  # n = 900
    
    # Input(X): Plackett-Luceで生成した市場予測
    market_prob_tensor = torch.zeros(n, 18, 3) # [900, 18, 3]
    # Output(y): 正解の着順 (1-3着)
    rank_tensor_one_hot = torch.zeros(n, 18, 3)  # [900, 18, 3]
    
    for i, (odds, ranks) in enumerate(zip(odds_list, rank_list)):
        # Plackett-Luceモデルで市場予測を生成
        pl_matrix = plackett_luce_model(odds)
        market_prob_tensor[i, :len(odds), :] = pl_matrix

        # 正解ラベルを作成 (1-3着)
        for horse_idx, rank in enumerate(ranks):
            if rank < 3:
                rank_tensor_one_hot[i, horse_idx, rank] = 1.0  # one-hot
    
    # 訓練・検証分割（8:2）
    split = int(n * 0.8)  # 720
    
    train_dataset = HorseRacingDataset(market_prob_tensor[:split], rank_tensor_one_hot[:split])
    test_dataset = HorseRacingDataset(market_prob_tensor[split:], rank_tensor_one_hot[split:])
    
    return train_dataset, test_dataset

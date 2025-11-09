import torch
import torch.nn as nn

class HorseRacingModel(nn.Module):
    """
    単勝オッズ → 二重確率行列
    
    入力: [batch, 18] 単勝オッズ
    出力: [batch, 18, 18] 二重確率行列
    """
    
    def __init__(self):
        super().__init__()
        channels = 1
        
        self.layers = nn.Sequential(
            nn.Linear(channels, 18),  # [batch, channels, 18, 18]
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Softmax(dim=2),
        )
    
    def forward(self, x):
        """
        Args:
            odds: [batch, 18] 単勝オッズ
        Returns:
            matrix: [batch, 18, 18] 二重確率行列
        """
        x = x.reshape(-1, 1, 18, 1)  # [batch, 1, 18, 1]
        x = self.layers(x)  # [batch, 1, 18, 18]
        x = x.squeeze()  # [batch, 18, 18]
        return x

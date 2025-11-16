import torch
import torch.nn as nn

class HorseRacingModel(nn.Module):
    """
    市場の1-3着予測確率行列 → 真の1-3着確率行列
    
    入力: [batch, 18, 3] 市場予測
    出力: [batch, 18, 3] 真の確率予測
    """
    
    def __init__(self):
        super().__init__()
        
        input_channels = 1
        hidden_channels = 4
        kernel_size = (3, 3)
        
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False
            ),  # [batch, 4, 18, 3]
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False
            ),  # [batch, 1, 18, 3]
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Softmax(dim=2),  # 着順ごとに各馬が該当する確率を正規化
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, 18, 3] 市場予測確率行列
        Returns:
            matrix: [batch, 18, 3] 補正後の確率行列
        """
        x = x.unsqueeze(1)  # [batch, 1, 18, 3]
        x = self.layers(x)  # [batch, 1, 18, 3]
        x = x.squeeze()  # [batch, 18, 3]
        return x

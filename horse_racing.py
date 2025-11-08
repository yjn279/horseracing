"""
競馬オッズ補正システム - 可変頭数対応

市場オッズから導出した確率行列を、ニューラルネットワークで補正し、
フェイバリット・ロングショットバイアスを学習する。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================================================================
# データ生成
# ============================================================================

def generate_data(n_races=2000, n_horses=10, bias=0.3):
    """
    ハードコーディングされたサンプルレースデータを生成
    
    固定のパターンを繰り返して必要なレース数を生成
    
    Args:
        n_races: 生成するレース数
        n_horses: 1レースあたりの出走頭数（8, 10, 12, 14, 16のみ対応）
        bias: 使用されない（互換性のため残す）
    
    Returns:
        market_matrix: 市場確率行列 [n_races, n_horses, n_horses]
        true_matrix: 真の確率行列 [n_races, n_horses, n_horses]
        rankings: 実際の着順 [n_races, n_horses]
    """
    # ハードコーディングされたサンプルパターン
    sample_data = _get_hardcoded_samples(n_horses)
    
    # パターンを繰り返して必要なレース数を生成
    n_patterns = len(sample_data['rankings'])
    n_repeats = (n_races + n_patterns - 1) // n_patterns
    
    true_matrix = np.tile(sample_data['true'], (n_repeats, 1, 1))[:n_races]
    market_matrix = np.tile(sample_data['market'], (n_repeats, 1, 1))[:n_races]
    rankings = np.tile(sample_data['rankings'], (n_repeats, 1))[:n_races]
    
    return market_matrix, true_matrix, rankings


def _get_hardcoded_samples(n_horses):
    """
    ハードコーディングされたサンプルデータを返す
    
    実際の競馬に近いパターンを手動で定義
    """
    if n_horses == 10:
        # 10頭立ての基本パターン（3レース分）
        
        # パターン1: 本命馬が明確（馬0が強い）
        true1 = np.array([
            [0.350, 0.250, 0.180, 0.100, 0.060, 0.030, 0.015, 0.010, 0.003, 0.002],  # 馬0: 強い
            [0.180, 0.200, 0.200, 0.150, 0.120, 0.080, 0.040, 0.020, 0.007, 0.003],  # 馬1
            [0.150, 0.180, 0.200, 0.170, 0.130, 0.090, 0.050, 0.020, 0.007, 0.003],  # 馬2
            [0.120, 0.140, 0.160, 0.180, 0.160, 0.120, 0.070, 0.035, 0.010, 0.005],  # 馬3
            [0.080, 0.100, 0.120, 0.150, 0.180, 0.160, 0.120, 0.060, 0.020, 0.010],  # 馬4
            [0.060, 0.070, 0.080, 0.110, 0.140, 0.180, 0.160, 0.120, 0.050, 0.030],  # 馬5
            [0.040, 0.040, 0.040, 0.080, 0.120, 0.160, 0.200, 0.180, 0.100, 0.040],  # 馬6
            [0.020, 0.020, 0.020, 0.060, 0.090, 0.130, 0.190, 0.240, 0.180, 0.050],  # 馬7
            [0.015, 0.015, 0.015, 0.040, 0.060, 0.090, 0.150, 0.250, 0.280, 0.085],  # 馬8
            [0.010, 0.010, 0.010, 0.030, 0.040, 0.050, 0.105, 0.165, 0.343, 0.237],  # 馬9: 弱い
        ])
        
        market1 = np.array([
            [0.400, 0.280, 0.170, 0.080, 0.040, 0.018, 0.008, 0.003, 0.001, 0.000],  # 馬0: 過大評価
            [0.190, 0.210, 0.200, 0.145, 0.115, 0.075, 0.038, 0.019, 0.006, 0.002],
            [0.155, 0.185, 0.200, 0.168, 0.128, 0.088, 0.048, 0.019, 0.006, 0.003],
            [0.118, 0.138, 0.158, 0.178, 0.158, 0.118, 0.070, 0.037, 0.018, 0.007],
            [0.075, 0.095, 0.118, 0.148, 0.178, 0.158, 0.118, 0.070, 0.028, 0.012],
            [0.055, 0.065, 0.078, 0.108, 0.138, 0.178, 0.158, 0.118, 0.070, 0.032],
            [0.032, 0.038, 0.038, 0.078, 0.118, 0.158, 0.198, 0.178, 0.118, 0.044],
            [0.015, 0.018, 0.018, 0.058, 0.088, 0.128, 0.188, 0.238, 0.198, 0.051],
            [0.010, 0.013, 0.013, 0.038, 0.058, 0.088, 0.148, 0.248, 0.298, 0.086],
            [0.005, 0.008, 0.008, 0.028, 0.038, 0.048, 0.103, 0.163, 0.341, 0.258],
        ])
        
        rankings1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 馬0が1着
        
        # パターン2: 混戦
        true2 = np.array([
            [0.120, 0.130, 0.140, 0.130, 0.120, 0.100, 0.090, 0.070, 0.060, 0.040],
            [0.125, 0.135, 0.140, 0.130, 0.115, 0.095, 0.085, 0.070, 0.060, 0.045],
            [0.115, 0.130, 0.145, 0.135, 0.120, 0.100, 0.085, 0.070, 0.060, 0.040],
            [0.110, 0.125, 0.140, 0.140, 0.125, 0.105, 0.090, 0.070, 0.055, 0.040],
            [0.105, 0.120, 0.135, 0.140, 0.130, 0.110, 0.095, 0.075, 0.055, 0.035],
            [0.100, 0.115, 0.130, 0.135, 0.135, 0.120, 0.100, 0.080, 0.055, 0.030],
            [0.095, 0.105, 0.120, 0.130, 0.135, 0.130, 0.110, 0.090, 0.060, 0.025],
            [0.090, 0.100, 0.110, 0.120, 0.130, 0.135, 0.125, 0.105, 0.065, 0.020],
            [0.085, 0.095, 0.100, 0.110, 0.120, 0.130, 0.135, 0.125, 0.080, 0.020],
            [0.080, 0.090, 0.095, 0.100, 0.110, 0.120, 0.130, 0.140, 0.105, 0.030],
        ])
        
        market2 = np.array([
            [0.130, 0.138, 0.143, 0.128, 0.118, 0.098, 0.088, 0.070, 0.062, 0.025],
            [0.133, 0.143, 0.143, 0.128, 0.113, 0.093, 0.083, 0.070, 0.062, 0.032],
            [0.122, 0.138, 0.148, 0.133, 0.118, 0.098, 0.083, 0.070, 0.062, 0.028],
            [0.117, 0.133, 0.143, 0.138, 0.123, 0.103, 0.088, 0.070, 0.058, 0.027],
            [0.111, 0.128, 0.138, 0.138, 0.128, 0.108, 0.093, 0.075, 0.058, 0.023],
            [0.105, 0.123, 0.133, 0.133, 0.133, 0.118, 0.098, 0.080, 0.058, 0.019],
            [0.099, 0.113, 0.128, 0.128, 0.133, 0.128, 0.108, 0.090, 0.063, 0.010],
            [0.093, 0.108, 0.118, 0.118, 0.128, 0.133, 0.123, 0.105, 0.068, 0.006],
            [0.087, 0.103, 0.108, 0.108, 0.118, 0.128, 0.133, 0.123, 0.083, 0.009],
            [0.082, 0.098, 0.103, 0.098, 0.108, 0.118, 0.128, 0.138, 0.108, 0.019],
        ])
        
        rankings2 = np.array([2, 0, 1, 3, 4, 5, 6, 7, 8, 9])  # 馬2が1着
        
        # パターン3: 穴馬（馬7）が来る
        true3 = np.array([
            [0.150, 0.170, 0.180, 0.150, 0.120, 0.090, 0.070, 0.040, 0.020, 0.010],
            [0.140, 0.165, 0.175, 0.155, 0.125, 0.095, 0.075, 0.042, 0.020, 0.008],
            [0.130, 0.155, 0.170, 0.160, 0.130, 0.100, 0.080, 0.045, 0.022, 0.008],
            [0.120, 0.145, 0.160, 0.165, 0.140, 0.110, 0.085, 0.048, 0.020, 0.007],
            [0.110, 0.135, 0.150, 0.165, 0.150, 0.125, 0.095, 0.045, 0.018, 0.007],
            [0.100, 0.125, 0.140, 0.155, 0.160, 0.140, 0.110, 0.045, 0.018, 0.007],
            [0.090, 0.110, 0.130, 0.145, 0.160, 0.160, 0.130, 0.050, 0.018, 0.007],
            [0.250, 0.220, 0.180, 0.130, 0.090, 0.060, 0.040, 0.020, 0.007, 0.003],  # 馬7: 実は強い
            [0.060, 0.070, 0.080, 0.095, 0.110, 0.130, 0.155, 0.180, 0.090, 0.030],
            [0.050, 0.060, 0.070, 0.080, 0.095, 0.110, 0.130, 0.165, 0.150, 0.090],
        ])
        
        market3 = np.array([
            [0.158, 0.178, 0.183, 0.148, 0.118, 0.088, 0.068, 0.038, 0.018, 0.003],
            [0.147, 0.173, 0.178, 0.153, 0.123, 0.093, 0.073, 0.040, 0.018, 0.002],
            [0.136, 0.163, 0.173, 0.158, 0.128, 0.098, 0.078, 0.043, 0.020, 0.003],
            [0.125, 0.153, 0.163, 0.163, 0.138, 0.108, 0.083, 0.046, 0.018, 0.003],
            [0.114, 0.143, 0.153, 0.163, 0.148, 0.123, 0.093, 0.043, 0.016, 0.004],
            [0.103, 0.133, 0.143, 0.153, 0.158, 0.138, 0.108, 0.043, 0.016, 0.005],
            [0.092, 0.118, 0.133, 0.143, 0.158, 0.158, 0.128, 0.048, 0.016, 0.006],
            [0.300, 0.250, 0.188, 0.128, 0.088, 0.058, 0.038, 0.018, 0.005, 0.002],  # 馬7: 過大評価
            [0.057, 0.068, 0.078, 0.093, 0.108, 0.128, 0.153, 0.178, 0.093, 0.044],
            [0.047, 0.058, 0.068, 0.078, 0.093, 0.108, 0.128, 0.163, 0.148, 0.109],
        ])
        
        rankings3 = np.array([7, 0, 1, 2, 3, 4, 5, 6, 8, 9])  # 馬7が1着（穴）
        
        # 3つのパターンを結合
        return {
            'true': np.array([true1, true2, true3]),
            'market': np.array([market1, market2, market3]),
            'rankings': np.array([rankings1, rankings2, rankings3])
        }
    
    else:
        # 他の頭数はランダム生成（簡易実装）
        np.random.seed(42)  # 再現性のため
        n_patterns = 3
        true_matrices = []
        market_matrices = []
        rankings_list = []
        
        for _ in range(n_patterns):
            strengths = np.random.gamma(2, 1, n_horses)
            strengths = strengths / strengths.sum()
            
            # 真の確率行列
            true_matrix = np.zeros((n_horses, n_horses))
            for h in range(n_horses):
                for r in range(n_horses):
                    true_matrix[h, r] = strengths[h] * np.exp(-strengths[h] * r * 2)
                true_matrix[h, :] /= true_matrix[h, :].sum()
            
            # 市場バイアス
            market_matrix = true_matrix.copy()
            for h in range(n_horses):
                fav = strengths[h] / strengths.mean()
                for r in range(n_horses):
                    w = 1.0 - 0.3 * fav * (1 - r / n_horses) if r < n_horses // 2 else 1.0 + 0.3 * fav * (r / n_horses - 0.5)
                    market_matrix[h, r] = true_matrix[h, r] ** w
                market_matrix[h, :] /= market_matrix[h, :].sum()
            
            # 着順
            rankings = np.zeros(n_horses, dtype=int)
            remaining = list(range(n_horses))
            for rank in range(n_horses):
                probs = strengths[remaining] / strengths[remaining].sum()
                winner = np.random.choice(len(remaining), p=probs)
                rankings[remaining[winner]] = rank
                remaining.pop(winner)
            
            true_matrices.append(true_matrix)
            market_matrices.append(market_matrix)
            rankings_list.append(rankings)
        
        return {
            'true': np.array(true_matrices),
            'market': np.array(market_matrices),
            'rankings': np.array(rankings_list)
        }


# ============================================================================
# ニューラルネットワークモデル
# ============================================================================

class OddsCorrector(nn.Module):
    """
    市場確率行列を補正するニューラルネットワーク
    
    可変頭数対応（8〜max_horses頭まで）
    各馬の順位確率分布を独立に処理し、バイアスを学習する
    """
    
    def __init__(self, max_horses=18, hidden_dim=128):
        """
        Args:
            max_horses: 対応する最大頭数
            hidden_dim: 隠れ層のニューロン数
        """
        super().__init__()
        self.max_horses = max_horses
        
        # 3層ニューラルネットワーク
        self.network = nn.Sequential(
            nn.Linear(max_horses, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, max_horses)
        )
    
    def forward(self, market_probs):
        """
        市場確率行列を補正
        
        Args:
            market_probs: [batch_size, n_horses, n_horses] 市場確率行列
        
        Returns:
            corrected_probs: [batch_size, n_horses, n_horses] 補正確率行列
        """
        batch_size, n_horses, n_ranks = market_probs.shape
        
        # 可変頭数対応: 最大頭数までパディング
        if n_ranks < self.max_horses:
            padding = torch.zeros(
                batch_size, n_horses, self.max_horses - n_ranks,
                device=market_probs.device
            )
            market_probs = torch.cat([market_probs, padding], dim=2)
        
        # 対数変換（数値安定性向上）
        log_probs = torch.log(market_probs + 1e-10)
        
        # 全馬を一括処理（行列演算）
        # [batch, n_horses, max_horses] → [batch * n_horses, max_horses]
        flat_input = log_probs.view(batch_size * n_horses, self.max_horses)
        flat_output = self.network(flat_input)
        output = flat_output.view(batch_size, n_horses, self.max_horses)
        
        # 元の頭数にトリミング & 確率正規化
        output = output[:, :, :n_ranks]
        corrected_probs = torch.softmax(output, dim=2)
        
        return corrected_probs


# ============================================================================
# 訓練・評価
# ============================================================================

def train_model(model, train_loader, val_loader, n_epochs=50, learning_rate=0.001):
    """
    モデルを訓練
    
    Args:
        model: OddsCorrectorモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        n_epochs: エポック数
        learning_rate: 学習率
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(n_epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        
        for market_probs, true_probs, _ in train_loader:
            optimizer.zero_grad()
            
            # 予測
            predicted_probs = model(market_probs)
            
            # 損失計算（KL Divergence）
            pred_log = torch.log(predicted_probs + 1e-10)
            loss = criterion(
                pred_log.view(-1, predicted_probs.shape[-1]),
                true_probs.view(-1, true_probs.shape[-1])
            )
            
            # 逆伝播 & パラメータ更新
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for market_probs, true_probs, _ in val_loader:
                predicted_probs = model(market_probs)
                pred_log = torch.log(predicted_probs + 1e-10)
                
                loss = criterion(
                    pred_log.view(-1, predicted_probs.shape[-1]),
                    true_probs.view(-1, true_probs.shape[-1])
                )
                val_loss += loss.item()
        
        # 進捗表示
        if (epoch + 1) % 10 == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")


def evaluate_model(model, market_matrix, rankings):
    """
    モデルの性能を評価
    
    Args:
        model: 訓練済みモデル
        market_matrix: 市場確率行列
        rankings: 実際の着順
    
    Returns:
        (market_acc, corrected_acc, market_ll, corrected_ll)
    """
    model.eval()
    
    # 補正確率を予測
    with torch.no_grad():
        corrected_matrix = model(torch.FloatTensor(market_matrix)).numpy()
    
    # 評価指標1: 1着的中率
    actual_winners = (rankings == 0).argmax(axis=1)  # 実際の1着馬
    market_predictions = market_matrix[:, :, 0].argmax(axis=1)
    corrected_predictions = corrected_matrix[:, :, 0].argmax(axis=1)
    
    market_accuracy = (market_predictions == actual_winners).mean()
    corrected_accuracy = (corrected_predictions == actual_winners).mean()
    
    # 評価指標2: Log Loss（全順位の確率予測精度）
    market_log_loss = _calculate_log_loss(market_matrix, rankings)
    corrected_log_loss = _calculate_log_loss(corrected_matrix, rankings)
    
    return market_accuracy, corrected_accuracy, market_log_loss, corrected_log_loss


def _calculate_log_loss(prob_matrix, rankings):
    """Log Lossを計算（全順位）"""
    eps = 1e-15
    log_probs = []
    
    for race_idx in range(len(prob_matrix)):
        for horse_idx in range(prob_matrix.shape[1]):
            actual_rank = rankings[race_idx, horse_idx]
            predicted_prob = prob_matrix[race_idx, horse_idx, actual_rank]
            log_probs.append(np.log(np.clip(predicted_prob, eps, 1.0)))
    
    return -np.mean(log_probs)


# ============================================================================
# メイン実行
# ============================================================================

def main():
    """メイン実行フロー"""
    print("=" * 60)
    print("競馬オッズ補正システム（可変頭数対応）")
    print("=" * 60)
    
    # 設定
    HORSE_COUNTS = [8, 10, 12, 14, 16]
    RACES_PER_COUNT = 1000
    TRAIN_HORSE_COUNT = 10  # 訓練する頭数
    MAX_HORSES = 18
    BATCH_SIZE = 64
    EPOCHS = 50
    
    # ステップ1: 複数の頭数でデータ生成
    print("\n[1/3] データ生成...")
    all_data = {}
    
    for n_horses in HORSE_COUNTS:
        market, true, rankings = generate_data(
            n_races=RACES_PER_COUNT,
            n_horses=n_horses,
            bias=0.3
        )
        all_data[n_horses] = (market, true, rankings)
        print(f"  {n_horses}頭立て: {RACES_PER_COUNT}レース生成")
    
    # ステップ2: モデル訓練（10頭立てデータで学習）
    print(f"\n[2/3] 訓練（{TRAIN_HORSE_COUNT}頭立てデータ）...")
    
    market_matrix, true_matrix, rankings = all_data[TRAIN_HORSE_COUNT]
    split_idx = int(0.8 * len(market_matrix))
    
    # データローダー作成
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(market_matrix[:split_idx]),
            torch.FloatTensor(true_matrix[:split_idx]),
            torch.LongTensor(rankings[:split_idx])
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(market_matrix[split_idx:]),
            torch.FloatTensor(true_matrix[split_idx:]),
            torch.LongTensor(rankings[split_idx:])
        ),
        batch_size=BATCH_SIZE
    )
    
    # モデル初期化 & 訓練
    model = OddsCorrector(max_horses=MAX_HORSES, hidden_dim=128)
    train_model(model, train_loader, val_loader, n_epochs=EPOCHS)
    
    # ステップ3: 複数の頭数で評価
    print("\n[3/3] 評価（全頭数）...")
    print(f"{'頭数':<6} {'市場的中':<10} {'補正的中':<10} {'市場LL':<10} {'補正LL':<10} {'改善':<10}")
    print("-" * 65)
    
    for n_horses in HORSE_COUNTS:
        market, _, rankings = all_data[n_horses]
        test_market = market[split_idx:]
        test_rankings = rankings[split_idx:]
        
        market_acc, corrected_acc, market_ll, corrected_ll = evaluate_model(
            model, test_market, test_rankings
        )
        
        improvement = market_ll - corrected_ll
        
        print(f"{n_horses}頭    {market_acc:.4f}     {corrected_acc:.4f}     "
              f"{market_ll:.4f}     {corrected_ll:.4f}     {improvement:+.4f}")
    
    print("\n" + "=" * 60)
    print("✓ 完了")


if __name__ == '__main__':
    main()

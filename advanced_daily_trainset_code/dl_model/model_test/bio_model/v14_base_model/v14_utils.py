import os
import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from advanced_daily_trainset_code.dl_model.model_test.bio_model.v14_base_model.v14_bio_clustering import generate_name_tag


# ++ ===================================================================================================================

# 시드 지정
def seed_everything(seed) :
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ++ ===================================================================================================================

# 감염(1) 데이터가 적으므로 더 자주 샘플링하도록 가중치 설정. 층화표본 가중치를 생성하여 층화표본 샘플링 적용
# 기본적으로 비복원 추출을 진행하지만, 감염(1) 데이터를 더 많이 미니배치에 포함시키기 위해 replacement = True로 복원 추출을 할 수도 있음
def weighted_random_sampler(x, replacement = False):
    x = pd.Series(x).astype(int)
    class_sample_count = np.array(x.value_counts().sort_index())
    weight = 1.0 / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in x]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement)
    return sampler

# ++ ===================================================================================================================

# Stratified Batch Sampler 구현 (각 배치에 일정 비율로 감염(1) 포함)
# 비복원 추출
class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels.astype(int))
        self.batch_size = batch_size

        # 감염 / 미감염 인덱스 분류
        self.infected_indices = np.where(self.labels == 1)[0] # 114
        self.non_infected_indices = np.where(self.labels == 0)[0] # 113471

        # 감염 비율 계산
        self.infected_ratio = len(self.infected_indices) / len(self.labels) # 114 / 113585
        self.non_infected_ratio = 1 - self.infected_ratio # 113471 / 1135885

        # 한 배치 내에서 감염 샘플 수 계산
        self.infected_batch_size = max(1, int(self.batch_size * self.infected_ratio)) # 최소 1개는 포함되도록
        self.non_infected_batch_size = self.batch_size - self.infected_batch_size # 그 나머지

        # 셔플링
        np.random.shuffle(self.infected_indices)
        np.random.shuffle(self.non_infected_indices)

    def __iter__(self):
        infected_perm = np.random.permutation(self.infected_indices)
        non_infected_perm = np.random.permutation(self.non_infected_indices)

        # 감염 샘플과 비감염 샘플을 기준으로 만들 수 있는 배치 개수 중 작은 값 선택
        # 감염 샘플이 적어서 감염 샘플을 기준으로 배치를 만들다가 부족해지면, 비감염 샘플이 남아 있어도 배치를 멈춰야 하기 때문
        min_batches = min(len(infected_perm) // self.infected_batch_size,
                          len(non_infected_perm) // self.non_infected_batch_size)

        for i in range(min_batches):
            infected_batch = infected_perm[
                             i * self.infected_batch_size:(i + 1) * self.infected_batch_size]
            non_infected_batch = non_infected_perm[
                                 i * self.non_infected_batch_size:(i + 1) * self.non_infected_batch_size]

            batch = np.concatenate([infected_batch, non_infected_batch])
            np.random.shuffle(batch)  # 배치 내 샘플 셔플링
            yield batch.tolist()

    def __len__(self):
        min_batches = min(len(self.infected_indices) // self.infected_batch_size,
                          len(self.non_infected_indices) // self.non_infected_batch_size)
        return min_batches

# ++ ===================================================================================================================

# Stratified Batch Sampler 구현 (각 배치에 일정 비율로 감염(1) 포함)
# 복원 추출
# class StratifiedBatchSampler(Sampler):
#     def __init__(self, labels, batch_size):
#         self.labels = np.array(labels)
#         self.batch_size = batch_size
#
#         # 감염 / 비감염 인덱스 분류
#         self.infected_indices = np.where(self.labels == 1)[0]  # 감염 샘플
#         self.non_infected_indices = np.where(self.labels == 0)[0]  # 비감염 샘플
#
#         # 감염 비율 계산
#         self.infected_ratio = len(self.infected_indices) / len(self.labels)
#         self.non_infected_ratio = 1 - self.infected_ratio
#
#         # 한 배치 내에서 감염 샘플 수 계산 (최소 2개 이상 포함되도록)
#         self.infected_batch_size = max(2, int(self.batch_size * self.infected_ratio))
#         self.non_infected_batch_size = self.batch_size - self.infected_batch_size
#
#     def __iter__(self):
#         for _ in range(len(self.labels) // self.batch_size):  # 전체 배치 개수 기준 반복
#             infected_batch = np.random.choice(self.infected_indices, self.infected_batch_size, replace=True)
#             non_infected_batch = np.random.choice(self.non_infected_indices, self.non_infected_batch_size, replace=True)
#
#             batch = np.concatenate([infected_batch, non_infected_batch])
#             np.random.shuffle(batch)  # 배치 내 셔플링
#             yield batch.tolist()
#
#     def __len__(self):
#         return len(self.labels) // self.batch_size  # 전체 데이터 크기를 기준으로 배치 개수 결정

# ++ ===================================================================================================================

class TverskyLoss(nn.Module):
    def __init__(self, alpha = 0.2, beta = 0.8, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):

        # Flatten label and inputsiction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        return 1 - Tversky

# ++ ===================================================================================================================

def evaluate_classification(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    return accuracy, precision, recall, f1, auroc, auprc

# ++ ===================================================================================================================

def generate_train_test_pairs(start_year=2020, end_year=2025):
    """
    기존 조합 + 각 연도별 모든 연도 예측 조합을 생성하는 함수
    (자기 자신 예측 포함)

    Args:
        start_year: 시작 연도
        end_year: 끝 연도 (포함)

    Returns:
        train_test_pairs: [(학습연도리스트, 예측연도리스트)] 형태의 리스트
    """
    train_test_pairs = []
    all_years = list(range(start_year, end_year + 1))

    # 1. 기존 조합: 2020-2023년 학습 → 2024-2025년 예측
    base_train_years = list(range(start_year, end_year - 1))  # 2020-2023
    base_test_years = list(range(end_year - 1, end_year + 1))  # 2024-2025
    train_test_pairs.append((base_train_years, base_test_years))
    # print(f"기존 조합: 학습 {base_train_years} → 예측 {base_test_years}")

    # 2. 각 연도별 모든 연도 예측 (자기 자신 포함)
    # print("\n=== 각 연도별 모든 연도 예측 조합 ===")
    for train_year in all_years:
        for pred_year in all_years:
            train_test_pairs.append(([train_year], [pred_year]))
            self_pred = "(자기예측)" if train_year == pred_year else ""
            # print(f"학습: [{train_year}] → 예측: [{pred_year}] {self_pred}")

    total_combinations = len(train_test_pairs)
    # print(f"\n총 조합 수: {total_combinations}개")
    # print(f"- 기존 조합: 1개")
    # print(f"- 연도별 조합: {6 * 6}개 (각 연도가 6개 연도 예측)")

    return train_test_pairs

# ++ ===================================================================================================================

# cross entropy 성능지표
def histogram_score_reverse_gradi(df, percent_step=10, decay_rate=0.85):
# decay_rate = 0.9 → 천천히 감소
# decay_rate = 0.5 → 매우 급격히 감소
# 0.7 ~ 0.85 사이가 일반적으로 많이 쓰임

    # 1. 날짜별 위험도 순위 및 퍼센트 계산
    df['risk_rank'] = df.groupby('standard_date')['prob'].rank(method='min', ascending=False)
    df['total_per_date'] = df.groupby('standard_date')['prob'].transform('count')
    df['risk_percent'] = (df['risk_rank'] / df['total_per_date']) * 100
    df['risk_percent'] = np.clip(df['risk_percent'], 0, 100)  # 안전하게 클립

    # 2. 실제 발생 농장만 추출
    df_ai = df[df['asf_occurrence_yn'] == 1].copy()
    real_positive_count = len(df_ai)

    # 3. 퍼센트 구간 정의
    percent_bins = np.arange(0, 101, percent_step)
    num_bins = len(percent_bins) - 1

    # 4. 이상적 분포 P: 지수적으로 감소하는 분포 생성
    ideal_manual = np.array([decay_rate ** i for i in range(num_bins)])
    P = ideal_manual / ideal_manual.sum()
    ideal_counts = P * real_positive_count  # 실제 개수 스케일 (optional)

    # 5. 실제 발생 농장의 퍼센트 기반 분포 Q 계산
    actual_counts, _ = np.histogram(df_ai['risk_percent'], bins=percent_bins)
    Q = actual_counts / actual_counts.sum()  # 확률분포로 정규화

    # 6. 크로스 엔트로피 계산
    cross_entropy = -np.sum(Q * np.log(1-P))

    # 7. 결과 테이블 생성
    hist_df = pd.DataFrame({
        'bin_range': [f'{percent_bins[i]}~{percent_bins[i+1]}%' for i in range(num_bins)],
        'ideal_count(P)': ideal_counts,
        'actual_count(Q)': actual_counts,
        'P (prob)': P,
        'Q (prob)': Q,
        '1-P (prob)': 1-P,
        'log(1-P)': np.log(1-P),
        'Q * log(1-P)': Q * np.log(1-P)
    })

    return cross_entropy

# ++ ===================================================================================================================

# 발생농장 위험도 히스토그램 시각화
def plot_risk_percent_histogram(df, prob, year_label, cluster_order, model_path, save_name):
    df = df.copy()
    df['prob'] = prob
    df['risk_rank'] = df.groupby('standard_date')['prob'].rank(method='min', ascending=False)
    df['total_per_date'] = df.groupby('standard_date')['prob'].transform('count')
    df['risk_percent'] = (df['risk_rank'] / df['total_per_date']) * 100
    df = df[df['asf_occurrence_yn'] == 1].copy()

    bins = list(range(0, 110, 5))  # 0, 5, ..., 105
    df['percent_bin'] = pd.cut(df['risk_percent'], bins=bins, right=False)
    hist = df['percent_bin'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    ax = hist.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title(f"{year_label} ASF 발생 농장의 위험도 순위 분포 (5% 구간)")
    plt.xlabel("위험도 상위 퍼센트 구간")
    plt.ylabel("발생 농가 수")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2., height + 0.3, int(height),
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(model_path, save_name), dpi=300, bbox_inches='tight')
    plt.close()

# 클러스터 절대위험도 박스플롯 시각화
def plot_cluster_boxplot(df, prob, cluster_order, model_path, save_name):
    df = df.copy()
    df['prob'] = prob
    df['cluster'] = df.apply(generate_name_tag, axis=1)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y='prob', data=df, palette='Set3', order=cluster_order)

    plt.title('클러스터별 위험도 확률 (prob) 분포', fontsize=14)
    plt.xlabel('클러스터', fontsize=12)
    plt.ylabel('예측 위험도 (prob)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, save_name), dpi=300, bbox_inches='tight')
    plt.close()

# ++ ===================================================================================================================

# 6 x 6 성능지표 테이블 생성
def create_metric_matrix(df, metric_col, model_path, years=[2020, 2021, 2022, 2023, 2024, 2025]):
    matrix = pd.DataFrame(np.nan, index=years, columns=years)

    for _, row in df.iterrows():
        train = row['train_year']
        test = row['test_year']
        metric = row.get(metric_col, np.nan)

        if isinstance(test, list):
            if len(test) != 1:
                continue
            test = test[0]

        if isinstance(train, str) and '-' in train:
            start, end = train.split('-')
            if start != end:
                continue
            train = int(start)

        try:
            train, test = int(train), int(test)
        except:
            continue

        if train in years and test in years:
            matrix.loc[train, test] = metric

    matrix.index = [f'Train_{y}' for y in matrix.index]
    matrix.columns = [f'Test_{y}' for y in matrix.columns]
    return matrix

# 6 x 6 성능지표 히트맵 시각화
def plot_metric_matrix(matrix_df, metric_name, model_path, cmap='YlOrRd'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_df, annot=True, fmt=".3f", cmap=cmap,
                mask=matrix_df.isna(), cbar_kws={'label': metric_name},
                square=True, linewidths=0.5)

    plt.title(f"{metric_name} Performance Matrix\n(행: 학습연도, 열: 예측연도)", fontsize=14, pad=20)
    plt.xlabel("예측 연도", fontsize=12)
    plt.ylabel("학습 연도", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    img_name = f"{metric_name.lower().replace(' ', '_')}_matrix_{timestamp}.png"
    csv_name = f"{metric_name.lower().replace(' ', '_')}_matrix_{timestamp}.csv"

    plt.savefig(os.path.join(model_path, img_name), dpi=300)
    matrix_df.to_csv(os.path.join(model_path, csv_name), encoding='utf-8')
    plt.close()
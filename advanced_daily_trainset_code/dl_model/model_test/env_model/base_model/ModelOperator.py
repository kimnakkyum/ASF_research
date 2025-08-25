import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from advanced_daily_trainset_code.dl_model.model_test.env_model.base_model.LoadData import get_input_data
from advanced_daily_trainset_code.dl_model.model_test.env_model.base_model.env_clustering import make_cluster
from advanced_daily_trainset_code.dl_model.model_test.env_model.base_model.env_clustering import generate_name_tag
from advanced_daily_trainset_code.dl_model.model_test.env_model.base_model.HDNN import HDNN
from advanced_daily_trainset_code.dl_model.model_test.env_model.base_model.model_config import model_config as mc
from advanced_daily_trainset_code.dl_model.model_test.env_model.base_model.utils import seed_everything, weighted_random_sampler, \
    StratifiedBatchSampler, TverskyLoss, evaluate_classification, generate_train_test_pairs, histogram_score_reverse_gradi, \
    plot_risk_percent_histogram, plot_cluster_boxplot, create_metric_matrix, plot_metric_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader

# ++ ===================================================================================================================

# 윈도우 기본 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 모델 경로 설정
model_path = "D:/주식회사빅밸류 Dropbox/김낙겸/19. 2025 업무/250101 농림축산검역본부/250428_ASF 분석 데이터 구축/모델 결과/환경 모델/v27_baseline_model_testestest"


# 모델 활용 변수 카테고리, 저장 경로 설정
train_col_cat = '환경'

# 데이터 로드
df = get_input_data(train_col_cat)

df.reset_index(drop=True, inplace=True)

# 군집분석 기반 변수 전처리
df_clustered = make_cluster(df)

# 학습셋 저장
file_path = os.path.join(model_path, "trainset.parquet")
directory = os.path.dirname(file_path)
os.makedirs(directory)
df_clustered.to_parquet(file_path)


# CUDA 활용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")


# 시드 고정
seed_everything(42)

# 학습 예측 정의 (20-23 -> 24-25)
train_test_pairs = []

train_years_combined = list(range(2020, 2024))  # 2020-2024
test_year_final = list(range(2024, 2026))
train_test_pairs.append((train_years_combined, test_year_final))

print("Train-Test 쌍들:")
for i, (train, test) in enumerate(train_test_pairs):
    print(f"{i+1}. Train: {train}, Test: {test}")


# # 학습-예측 조합 생성
# train_test_pairs = generate_train_test_pairs(start_year=2020, end_year=2025)

# 전체 결과를 저장할 리스트
all_results = []

df_clustered['year'] = pd.to_datetime(df_clustered['standard_date']).dt.year


# 연도별 교차 검증 수행
for train_year, test_year in train_test_pairs:
    print(f"\n========== {train_year}년 학습 -> {test_year}년 테스트 ==========")

    # 연도별 데이터 분할
    if isinstance(train_year, list):
        # 여러 연도를 합쳐서 학습 (2020-2024년 전체 -> 2025년 케이스)
        train_df = df_clustered[df_clustered['year'].isin(train_year)].copy()
        train_year_str = f"{min(train_year)}-{max(train_year)}"
    else:
        # 단일 연도 학습
        train_df = df_clustered[df_clustered['year'] == train_year].copy()
        train_year_str = str(train_year)

    test_df = df_clustered[df_clustered['year'].isin(test_year)].copy()

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # numpy 배열로 변환
    train_numpy = train_df.iloc[:, 2:-1].to_numpy()
    test_numpy = test_df.iloc[:, 2:-1].to_numpy()

    # 모델 경로 설정
    os.makedirs(model_path, exist_ok=True)
    model_name_prefix = f'{train_year_str}_to_{test_year}'

    # 모델 구조 (매번 새로 초기화)
    model = HDNN().to(device)

    # 손실 함수, 학습률, 옵티마이저, 스케쥴러 정의
    loss = TverskyLoss()
    learning_rate = mc.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch,
                                                  last_epoch=-1)

    # 하이퍼파라미터 지정
    epoch = mc.epoch
    patience = mc.patience
    best_optim_loss = np.inf
    patience_check = 0

    # 학습 및 검증 성능 list
    train_tversky_loss_list = []
    train_accuracy_list, train_precision_list, train_recall_list, train_f1_list, train_auroc_list, train_auprc_list, train_recall_top10_list, train_cross_entropy_list, train_risk_percent_max_list = [], [], [], [], [], [], [], [], []
    test_tversky_loss_list = []
    test_accuracy_list, test_precision_list, test_recall_list, test_f1_list, test_auroc_list, test_auprc_list, test_recall_top10_list, test_cross_entropy_list, test_risk_percent_max_list = [], [], [], [], [], [], [], [], []

    # 학습 루프
    for n in tqdm(range(epoch), desc=f'{train_year_str}->{test_year}'):

        # 모델 학습
        model.train()

        # 미니 배치 없이 가중치 업데이트
        train_tensor = torch.tensor(train_numpy, dtype=torch.float32).to(device)
        x_train, y_train = train_tensor[:, 1:], train_tensor[:, 0]

        # 누적된 그래디언트 초기화
        optimizer.zero_grad()

        # 예측 값 산출
        y_prob = model(x_train)

        # tversky Loss 계산
        tversky_loss = loss(y_prob, y_train)

        # 역전파를 통해 그래디언트 계산
        tversky_loss.backward()

        # 가중치 업데이트
        optimizer.step()

        # 스케쥴러 업데이트
        scheduler.step()

        # 실제 값, 예측 값, 예측 확률
        y_true = y_train.to('cpu').detach().numpy()
        y_prob_cpu = y_prob.to('cpu').detach().numpy()
        y_pred = y_prob_cpu.round()

        # Train 성능 계산
        train_tversky_loss_list.append(tversky_loss)
        accuracy, precision, recall, f1, auroc, auprc = evaluate_classification(y_true, y_pred, y_prob_cpu)
        train_accuracy_list.append(accuracy)
        train_precision_list.append(precision)
        train_recall_list.append(recall)
        train_f1_list.append(f1)
        train_auroc_list.append(auroc)
        train_auprc_list.append(auprc)

        # + 전체 감염 중 위험도 상위 10% 내 포함 비율
        train_df_copy = train_df.copy()
        train_prob = y_prob.to('cpu').detach().numpy()
        train_df_copy['prob'] = train_prob
        train_df_copy['risk_rank'] = train_df_copy.groupby('standard_date')['prob'].rank(method='min', ascending=False)
        train_df_copy['total_per_date'] = train_df_copy.groupby('standard_date')['prob'].transform('count')
        train_df_copy['risk_percent'] = (train_df_copy['risk_rank'] / train_df_copy['total_per_date']) * 100
        train_df_copy['risk_percent'] = np.clip(train_df_copy['risk_percent'], 0, 100)  # 안전하게 클립
        train_top10_count = len(train_df_copy[(train_df_copy['risk_percent'] < 10) & (train_df_copy['asf_occurrence_yn'] == 1)])
        train_asf_count = len(train_df_copy[train_df_copy['asf_occurrence_yn'] == 1])
        train_recall_top10 = train_top10_count / train_asf_count
        train_recall_top10_list.append(train_recall_top10)

        # 상대 위험도 구간별 개수, 크로스 엔트로피 계산
        train_cross_entropy = histogram_score_reverse_gradi(train_df_copy)
        train_cross_entropy_list.append(train_cross_entropy)

        # + 전체 감염 중 위험도 최솟값
        train_risk_percent_max = train_df_copy['risk_percent'].max()
        train_risk_percent_max_list.append(train_risk_percent_max)

        # 모델 검증
        model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(test_numpy, dtype=torch.float32).to(device)
            x_test, y_test = test_tensor[:, 1:], test_tensor[:, 0]

            test_true = y_test.cpu().numpy()
            test_prob = model(x_test).cpu().numpy()
            test_pred = test_prob.round()

            # test 성능 계산
            test_tversky_loss = loss(torch.tensor(test_prob, dtype=torch.float32).to(device),
                                     torch.tensor(test_true, dtype=torch.float32).to(device))
            test_tversky_loss_list.append(test_tversky_loss)

            test_accuracy, test_precision, test_recall, test_f1, test_auroc, test_auprc = evaluate_classification(
                test_true, test_pred, test_prob)
            test_accuracy_list.append(test_accuracy)
            test_precision_list.append(test_precision)
            test_recall_list.append(test_recall)
            test_f1_list.append(test_f1)
            test_auroc_list.append(test_auroc)
            test_auprc_list.append(test_auprc)

            # 전체 감염 중 위험도 상위 10% 내 포함 비율
            test_df_copy = test_df.copy()
            test_df_copy['prob'] = test_prob
            test_df_copy['risk_rank'] = test_df_copy.groupby('standard_date')['prob'].rank(method='min', ascending=False)
            test_df_copy['total_per_date'] = test_df_copy.groupby('standard_date')['prob'].transform('count')
            test_df_copy['risk_percent'] = (test_df_copy['risk_rank'] / test_df_copy['total_per_date']) * 100
            test_df_copy['risk_percent'] = np.clip(test_df_copy['risk_percent'], 0, 100)  # 안전하게 클립
            test_top10_count = len(test_df_copy[(test_df_copy['risk_percent'] < 10) & (test_df_copy['asf_occurrence_yn'] == 1)])
            test_asf_count = len(test_df_copy[test_df_copy['asf_occurrence_yn'] == 1])
            test_recall_top10 = test_top10_count/test_asf_count
            test_recall_top10_list.append(test_recall_top10)

            # 상대 위험도 구간별 개수, 크로스 엔트로피 계산
            test_cross_entropy = histogram_score_reverse_gradi(test_df_copy)
            test_cross_entropy_list.append(test_cross_entropy)

            # + 전체 감염 중 위험도 최솟값
            test_risk_percent_max = test_df_copy['risk_percent'].max()
            test_risk_percent_max_list.append(test_risk_percent_max)

        # 로그 출력
        model_log = (
            f'epoch :: {n + 1}, train_tversky_loss :: {tversky_loss:.4f}, train_precision :: {precision:.4f}, '
            f'train_recall :: {recall:.4f}, train_f1_score :: {f1:.4f}, train_auroc :: {auroc:.4f}, train_auprc :: {auprc:.4f} \n'
            f'test_tversky_loss :: {test_tversky_loss:.4f}, test_precision :: {test_precision:.4f}, test_recall :: {test_recall:.4f}, '
            f'test_f1_score :: {test_f1:.4f}, test_auroc :: {test_auroc:.4f}, test_auprc :: {test_auprc:.4f}, '
            f'test_recall_top10 :: {test_recall_top10:.4f}', f'test_cross_entropy :: {test_cross_entropy:.4f}', f'test_risk_percent_max :: {test_risk_percent_max:.4f}')

        if len(pd.DataFrame(test_pred).value_counts()) > 1:
            test_cnt = [pd.DataFrame(test_pred).value_counts()[0], pd.DataFrame(test_pred).value_counts()[1]]
        else:
            test_cnt = [pd.DataFrame(test_pred).value_counts().values[0]]

        print(
            f"Epoch {n + 1}: prob_range=[{test_prob.min():.4f}, {test_prob.max():.4f}], "
            f"pred_counts={test_cnt}, recall={test_recall:.4f}, f1={test_f1:.4f}, "
            f"tversky_loss={test_tversky_loss:.4f}, top10_recall={test_recall_top10:.4f}, "
            f"cross_entropy={test_cross_entropy:.4f}, risk_percent_max={test_risk_percent_max:.4f}"
        )

        # 최적 모델 저장
        if tversky_loss < best_optim_loss:
            print(f'New best model saved at epoch {n + 1}')
            best_optim_loss = tversky_loss
            # torch.save(model.state_dict(), f'{model_path}/best_optim_model_{train_year_str}_to_{test_year}.pt')
            # with open(f'{model_path}/best_optim_model_{train_year_str}_to_{test_year}.txt', 'w') as f:
            #     f.write(model_log)
            patience_check = 0
        else:
            patience_check += 1
            if patience_check >= patience:
                print(f'Early stopping at epoch {n + 1}')
                break

    # 최종 성능 저장
    final_result = {
        'train_year': train_year_str,
        'test_year': test_year,
        'final_test_precision': test_precision_list[-1],
        'final_test_recall': test_recall_list[-1],
        'final_test_f1': test_f1_list[-1],
        'final_test_auroc': test_auroc_list[-1],
        'final_test_auprc': test_auprc_list[-1],
        'final_test_recall_top10': test_recall_top10_list[-1],
        'test_risk_percent_max': test_risk_percent_max_list[-1],
        'final_test_tversky_loss': test_tversky_loss_list[-1].to('cpu').detach().numpy(),
        'final_test_cross_entropy': test_cross_entropy_list[-1],
        'train_data_size': len(train_df),
        'test_data_size': len(test_df),
        'train_positive_rate': train_df['asf_occurrence_yn'].mean(),
        'test_positive_rate': test_df['asf_occurrence_yn'].mean()
    }
    all_results.append(final_result)

    # 각 연도별 성능 데이터 저장
    train_test_metric_df = pd.DataFrame({
        'epochs': list(range(1, len(train_tversky_loss_list) + 1)),
        'train_tversky_loss': [x.to('cpu').detach().numpy() for x in train_tversky_loss_list],
        'train_precision': train_precision_list,
        'train_recall': train_recall_list,
        'train_f1_score': train_f1_list,
        'train_auroc': train_auroc_list,
        'train_auprc': train_auprc_list,
        'train_recall_top10' : train_recall_top10_list,
        'train_cross_entropy': train_cross_entropy_list,
        'train_risk_percent_max': train_risk_percent_max_list,
        'test_tversky_loss': [x.to('cpu').detach().numpy() for x in test_tversky_loss_list],
        'test_precision': test_precision_list,
        'test_recall': test_recall_list,
        'test_f1_score': test_f1_list,
        'test_auroc': test_auroc_list,
        'test_auprc': test_auprc_list,
        'test_recall_top10': test_recall_top10_list,
        'test_cross_entropy': test_cross_entropy_list,
        'test_risk_percent_max': test_risk_percent_max_list,
    })

    train_test_metric_df.to_csv(f'{model_path}/train_test_metric_df_{train_year_str}_to_{test_year}.txt', encoding='utf-8', sep='|', index=False)

    # cluster_order 사전 정의
    train_df_final = train_df.copy()
    test_df_final = test_df.copy()
    train_df_final['cluster'] = train_df_final.apply(generate_name_tag, axis=1)
    test_df_final['cluster'] = test_df_final.apply(generate_name_tag, axis=1)
    cluster_order = sorted(set(train_df_final['cluster'].unique()) | set(test_df_final['cluster'].unique()))

    # 학습 성능 시각화 및 저장
    plot_risk_percent_histogram(train_df, train_prob,f"{train_year_str}_to_{test_year}", cluster_order, model_path,f"train_hist_{train_year_str}.png")
    plot_cluster_boxplot(train_df, train_prob, cluster_order, model_path, f"test_cluster_boxplot_{train_year_str}.png")

    # 테스트 성능 시각화 및 저장
    plot_risk_percent_histogram(test_df, test_prob, f"{train_year_str}_to_{test_year}", cluster_order, model_path, f"test_hist_{train_year_str}_to_{test_year}.png")
    plot_cluster_boxplot(test_df, test_prob, cluster_order, model_path, f"test_cluster_boxplot_{train_year_str}_to_{test_year}.png")

    print(f"\n{train_year_str}->{test_year} 완료!")
    print(f"최종 성능: Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, "
          f"AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f}, Top10_Recall={test_recall_top10:.4f}, cross_entropy={test_cross_entropy:.4f}")

    train_df_final['prob'] = train_prob
    train_df_final['risk_rank'] = train_df_final.groupby('standard_date')['prob'].rank(method='min', ascending=False)
    train_df_final['total_per_date'] = train_df_final.groupby('standard_date')['prob'].transform('count')
    train_df_final['risk_percent'] = (train_df_final['risk_rank'] / train_df_final['total_per_date']) * 100

    test_df_final['prob'] = test_prob
    test_df_final['risk_rank'] = test_df_final.groupby('standard_date')['prob'].rank(method='min', ascending=False)
    test_df_final['total_per_date'] = test_df_final.groupby('standard_date')['prob'].transform('count')
    test_df_final['risk_percent'] = (test_df_final['risk_rank'] / test_df_final['total_per_date']) * 100

    train_df_final.to_parquet(os.path.join(model_path, "train_result.parquet"))
    test_df_final.to_parquet(os.path.join(model_path, "test_result.parquet"))

# 전체 결과 요약
print("\n========== 전체 결과 요약 ==========")
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))



# # Recall Top10 매트릭스
# recall_matrix = create_metric_matrix(results_df, 'final_test_recall_top10', model_path)
# plot_metric_matrix(recall_matrix, 'Test Recall Top10', model_path)
#
# # Cross Entropy 매트릭스
# ce_matrix = create_metric_matrix(results_df, 'final_test_cross_entropy', model_path)
# plot_metric_matrix(ce_matrix, 'Test Cross Entropy', model_path, cmap='Blues')
#
# # 상대 위험도 최솟값 매트릭스
# rm_matrix = create_metric_matrix(results_df, 'test_risk_percent_max', model_path)
# plot_metric_matrix(rm_matrix, 'Test Risk Percent_Max', model_path, cmap='coolwarm')




# 학습 클러스터별 예측값
train = pd.read_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\모델 결과\환경 모델\v27_baseline_model_test\train_result.parquet")
tot = (
    train.groupby('cluster')
    .agg(
        count=('prob', 'count'),
        median=('prob', 'median'),
        asf_occurrence_count=('asf_occurrence_yn', 'sum')  # 1의 개수
    )
    .reset_index()
)

tot.to_csv(os.path.join(model_path, "train_prob.csv"), encoding='cp949', index=False)



# 검증 클러스터별 예측값
test = pd.read_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\모델 결과\환경 모델\v27_baseline_model_test\test_result.parquet")
tot = (
    test.groupby('cluster')
    .agg(
        count=('prob', 'count'),
        median=('prob', 'median'),
        asf_occurrence_count=('asf_occurrence_yn', 'sum')  # 1의 개수
    )
    .reset_index()
)

tot.to_csv(os.path.join(model_path, "test_prob.csv"), encoding='cp949', index=False)





# # 학습기간 발생 데이터 정보 출력
# train = pd.read_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\모델 결과\환경 모델\v27_baseline_model_test\train_result.parquet")
#
# asf_org = train[train['asf_occurrence_yn'] == 1]
#
# train_asf_org = asf_org[['standard_date' ,'farm_serial_no', 'risk_percent', 'cluster']]
#
# train_asf_org.to_csv(os.path.join(model_path, "train_asf_org_risk_percent.csv"), encoding='cp949', index=False)
#
#
#
#
#
#
# # 검증기간 발생 데이터 정보 출력
# test = pd.read_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\모델 결과\환경 모델\v27_baseline_model_test\test_result.parquet")
#
# asf_org = test[test['asf_occurrence_yn'] == 1]
#
# test_asf_org = asf_org[['standard_date' ,'farm_serial_no', 'risk_percent', 'cluster']]
#
# test_asf_org.to_csv(os.path.join(model_path, "test_asf_org_risk_percent.csv"), encoding='cp949', index=False)
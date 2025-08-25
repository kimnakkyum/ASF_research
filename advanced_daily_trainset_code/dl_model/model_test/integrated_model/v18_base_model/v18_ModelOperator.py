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


from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_env_clustering import make_env_cluster
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_visit_clustering import make_visit_cluster
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_bio_clustering import make_bio_cluster
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_visit_clustering import visit_generate_name_tag
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_env_clustering import env_generate_name_tag
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_bio_clustering import bio_generate_name_tag

from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_HDNN import HDNN
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_model_config import model_config as mc
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_utils import seed_everything, weighted_random_sampler, \
    StratifiedBatchSampler, TverskyLoss, evaluate_classification, generate_train_test_pairs, histogram_score_reverse_gradi, \
    plot_risk_percent_histogram, plot_cluster_boxplot, create_metric_matrix, plot_metric_matrix, plot_sorted_scatter_pos_ties

import torch
from torch import nn
from torch.utils.data import DataLoader

# ++ ===================================================================================================================


# 윈도우 기본 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 모델 경로 설정
model_path = "D:/주식회사빅밸류 Dropbox/김낙겸/19. 2025 업무/250101 농림축산검역본부/250428_ASF 분석 데이터 구축/모델 결과/통합 모델/v18_baseline_model_whole_year"


# 모델 활용 변수 카테고리, 저장 경로 설정

# 데이터 로드
df = get_input_data()

# 환경 군집분석 기반 변수 전처리
env_clustered = make_env_cluster(df)

# 방역 군집분석 기반 변수 전처리
bio_clustered = make_bio_cluster(df)

# 전파 군집분석 기반 변수 전처리
visit_clustered = make_visit_cluster(df)

df_clustered = env_clustered.merge(bio_clustered).merge(visit_clustered)

df_clustered['asf_occurrence_yn'].sum()


# df_clustered = df_clustered.merge(base_df[['standard_date', 'farm_serial_no', 'farm_coordinate']])
#
#
# df_clustered = gpd.GeoDataFrame(df_clustered, geometry='farm_coordinate', crs="EPSG:5179")
#
# df_clustered.to_csv(r"C:\Users\BV-KIMNAKKYUM\Downloads\asf_df_clustered.csv", encoding='cp949', index=False)
#
# # GeoParquet으로 저장
# df_clustered.to_parquet(
#     r"C:\Users\BV-KIMNAKKYUM\Downloads\asf_df_clustered.parquet",
#     engine="pyarrow",  # GeoParquet은 pyarrow 필요
#     index=False
# )


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

# # 학습 예측 정의 (20-23 -> 24-25)
# train_test_pairs = []
#
# train_years_combined = list(range(2020, 2024))  # 2020-2024
# test_year_final = list(range(2024, 2026))
# train_test_pairs.append((train_years_combined, test_year_final))
#
# print("Train-Test 쌍들:")
# for i, (train, test) in enumerate(train_test_pairs):
#     print(f"{i+1}. Train: {train}, Test: {test}")


# # 학습-예측 조합 생성
# train_test_pairs = generate_train_test_pairs(start_year=2020, end_year=2025)

# 학습: [20-21], [20-21~21-22], [20-21~22-23], [20-21~23-24]
train_windows = [
    [2020],
    [2020, 2021],
    [2020, 2021, 2022],
    [2020, 2021, 2022, 2023],
    [2020, 2021, 2022, 2023, 2024]
]

test_seasons = [
    [2020], [2021], [2022], [2023], [2024], [2025]
]

train_test_pairs = [(tr, te) for tr in train_windows for te in test_seasons]




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

    cluster_funcs = {
        "cluster_env": env_generate_name_tag,
        "cluster_bio": bio_generate_name_tag,
        "cluster_visit": visit_generate_name_tag
    }

    # 모든 클러스터 생성
    for col_name, func in cluster_funcs.items():
        train_df_final[col_name] = train_df_final.apply(func, axis=1)
        test_df_final[col_name] = test_df_final.apply(func, axis=1)

    # 시각화 및 저장
    for col_name in cluster_funcs.keys():
        cluster_order = sorted(set(train_df_final[col_name].unique()) | set(test_df_final[col_name].unique()))
        plot_cluster_boxplot(
            train_df_final,
            train_prob,
            col_name,  # x축 컬럼명
            cluster_order,
            model_path,
            f"train_{col_name}_boxplot_{train_year_str}.png"
        )

    # 시각화 및 저장
    for col_name in cluster_funcs.keys():
        cluster_order = sorted(set(train_df_final[col_name].unique()) | set(test_df_final[col_name].unique()))
        plot_cluster_boxplot(
            test_df_final,
            test_prob,
            col_name,  # x축 컬럼명
            cluster_order,
            model_path,
            f"test_{col_name}_boxplot_{train_year_str}_to_{test_year}.png"
        )


    # train_df_final['env_cluster'] = train_df_final.apply(env_generate_name_tag, axis=1)
    # test_df_final['env_cluster'] = test_df_final.apply(env_generate_name_tag, axis=1)

    # cluster_order = sorted(set(train_df_final['cluster'].unique()) | set(test_df_final['cluster'].unique()))


    # 학습 성능 시각화 및 저장
    plot_risk_percent_histogram(train_df, train_prob,f"{train_year_str}_to_{test_year}", model_path,f"train_hist_{train_year_str}.png")
    # plot_cluster_boxplot(train_df, train_prob, cluster_order, model_path, f"test_cluster_boxplot_{train_year_str}.png")
    plot_sorted_scatter_pos_ties(train_df_copy, f"{train_year_str}", f"{train_year_str}", model_path)

    # 테스트 성능 시각화 및 저장
    plot_risk_percent_histogram(test_df, test_prob, f"{train_year_str}_to_{test_year}", model_path, f"test_hist_{train_year_str}_to_{test_year}.png")
    # plot_cluster_boxplot(test_df, test_prob, cluster_order, model_path, f"test_cluster_boxplot_{train_year_str}_to_{test_year}.png")
    plot_sorted_scatter_pos_ties(test_df_copy, f"{train_year_str}", f"{test_year}", model_path)

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



#
# # 학습 클러스터별 예측값
# train = pd.read_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\모델 결과/통합 모델/v18_baseline_model_test\train_result.parquet")
#
# # 2) 클러스터 라벨 생성 (중복 적용 제거)
# cluster_funcs = {
#     "cluster_env": env_generate_name_tag,
#     "cluster_bio": bio_generate_name_tag,
#     "cluster_visit": visit_generate_name_tag
# }
#
# for col_name, func in cluster_funcs.items():
#     if col_name not in train.columns:  # 이미 있으면 재생성 안 함(선택)
#         train[col_name] = train.apply(func, axis=1)
#
# # 3) 타입 정리(선택)
# train["standard_date"] = pd.to_datetime(train["standard_date"], errors="coerce")
#
#
#
# # 4) asf_occurrence_yn == 1 만 필터 + 필요한 컬럼만 선택
# cols = ["standard_date", "farm_serial_no", "risk_percent", "cluster_env", "cluster_bio", "cluster_visit"]
# out = (
#     train.loc[train["asf_occurrence_yn"] == 1, cols]
#          .sort_values(["farm_serial_no", "standard_date"])
#          .reset_index(drop=True)
# )
#
# out.to_csv(os.path.join(model_path, "train_prob.csv"), encoding='cp949', index=False)
#
#
#
#
# # 검증 클러스터별 예측값
# test = pd.read_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\모델 결과/통합 모델/v18_baseline_model_test\test_result.parquet")
#
# for col_name, func in cluster_funcs.items():
#     if col_name not in test.columns:  # 이미 있으면 재생성 안 함(선택)
#         test[col_name] = test.apply(func, axis=1)
#
# # 3) 타입 정리(선택)
# test["standard_date"] = pd.to_datetime(test["standard_date"], errors="coerce")
#
# # 4) asf_occurrence_yn == 1 만 필터 + 필요한 컬럼만 선택
# cols = ["standard_date", "farm_serial_no", "risk_percent", "cluster_env", "cluster_bio", "cluster_visit"]
# out = (
#     test.loc[test["asf_occurrence_yn"] == 1, cols]
#          .sort_values(["farm_serial_no", "standard_date"])
#          .reset_index(drop=True)
# )
#
# out.to_csv(os.path.join(model_path, "test_prob.csv"), encoding='cp949', index=False)








def _normalize_years(v):
    """train_year/test_year 값을 연도 리스트로 정규화."""
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        years = [int(x) for x in v]
    else:
        s = str(v)
        if "-" in s:  # "2020-2023" 같은 범위 문자열
            a, b = map(int, s.split("-"))
            years = list(range(min(a, b), max(a, b) + 1))
        else:
            years = [int(s)]
    return sorted(set(years))

def make_train_year_label(v):
    """[2020,2021,2022] -> '2020~2022 학습', 2020 -> '2020 학습'"""
    yrs = _normalize_years(v)
    return f"{yrs[0]} 학습" if len(yrs) == 1 else f"{yrs[0]}~{yrs[-1]} 학습"

def make_test_year_label(v):
    """테스트는 연도 단일 컬럼을 권장. (여러 해면 '시작~끝'으로 압축)"""
    yrs = _normalize_years(v)
    return f"{yrs[0]}" if len(yrs) == 1 else f"{yrs[0]}~{yrs[-1]}"

def create_metric_pivot_year(results_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """행=학습 연도 구간, 열=테스트 연도(또는 연도 범위), 값=metric_col"""
    df = results_df.copy()
    df["train_label"] = df["train_year"].apply(make_train_year_label)
    df["test_label"]  = df["test_year"].apply(make_test_year_label)

    pv = df.pivot(index="train_label", columns="test_label", values=metric_col)

    # 정렬: 열(테스트 연도 오름차순), 행(학습 시작/끝 오름차순)
    def col_key(s: str) -> int:
        s = str(s)
        start = int(s.split("~")[0])
        return start

    def row_key(s: str):
        s = str(s).replace(" 학습", "")
        parts = s.split("~")
        start = int(parts[0])
        end = int(parts[-1]) if len(parts) > 1 else start
        return (start, end)

    pv = pv.reindex(columns=sorted(pv.columns, key=col_key))
    pv = pv.reindex(index=sorted(pv.index, key=row_key))
    return pv





def plot_metric_heatmap(matrix: pd.DataFrame,
                        title: str,
                        save_dir: str,
                        filename: str,
                        as_percent: bool = False,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        cmap: str = "YlOrRd") -> None:   # <- cmap 추가
    os.makedirs(save_dir, exist_ok=True)
    data = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.6*matrix.shape[1] + 2,
                                    0.9*matrix.shape[0] + 2))
    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)  # <- cmap 적용
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title, rotation=90)

    ax.set_xticks(np.arange(matrix.shape[1])); ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0])); ax.set_yticklabels(matrix.index)
    ax.set_xlabel("테스트 연도"); ax.set_ylabel("모델 학습 기간(연도)"); ax.set_title(title, pad=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iloc[i, j]
            if pd.isna(v):
                continue
            txt = f"{v*100:.0f}%" if as_percent else f"{v:.5f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)




# 상위 10% 재현율 히트맵
recall_pivot = create_metric_pivot_year(results_df, "final_test_recall_top10")
plot_metric_heatmap(recall_pivot,
                    title="Recall@Top10 (Year-by-Year)",
                    save_dir=model_path,
                    filename="heatmap_recall_top10_year.png",
                    as_percent=False, vmin=0.0, vmax=1.0)

# 크로스 엔트로피 히트맵
ce_pivot = create_metric_pivot_year(results_df, "final_test_cross_entropy")
ce_pivot = np.round(ce_pivot, 3)
plot_metric_heatmap(ce_pivot,
                    title="Cross Entropy (Year-by-Year)",
                    save_dir=model_path,
                    filename="heatmap_cross_entropy_year.png",
                    as_percent=False)



# # Recall Top10 매트릭스
# recall_matrix = create_metric_matrix(results_df, 'final_test_recall_top10', model_path)
# plot_metric_matrix(recall_matrix, 'Test Recall Top10', model_path)
#
# # Cross Entropy 매트릭스
# ce_matrix = create_metric_matrix(results_df, 'final_test_cross_entropy', model_path)
# plot_metric_matrix(ce_matrix, 'Test Cross Entropy', model_path, cmap='Blues')
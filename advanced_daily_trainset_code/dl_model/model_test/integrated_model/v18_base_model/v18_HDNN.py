import torch
import torch.nn as nn

import pandas as pd
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_model_config import model_config

# 모델 구조
class HDNN(nn.Module):
    def __init__(self):
        super(HDNN, self).__init__()

        # model config
        self.mc = model_config()

        # 전체 피처 목록 및 인덱스 확보
        self.col_list = pd.DataFrame({'col': list(self.mc.env_col + self.mc.bio_col + self.mc.bio_mother_pig_col + self.mc.visit_col)})
        self.idx_full_list = list(self.col_list[self.col_list.col.isin(self.mc.env_col + self.mc.bio_col + self.mc.bio_mother_pig_col + self.mc.visit_col)].index)
        self.mc.pred_dim = len(self.idx_full_list)

        # ++ ===========================================================================================================

        # 축약 Vector Concat 후 FCNN 레이어 정의
        self.pred_nn_1 = nn.Linear(self.mc.pred_dim,
                                   self.mc.pred_dim * self.mc.pred_nn_1_multiply_num)
        self.pred_nn_1_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_1_multiply_num)

        self.pred_nn_2 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_1_multiply_num,
                                   self.mc.pred_dim * self.mc.pred_nn_2_multiply_num)
        self.pred_nn_2_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_2_multiply_num)

        self.pred_nn_3 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_2_multiply_num,
                                   self.mc.pred_dim * self.mc.pred_nn_3_multiply_num)
        self.pred_nn_3_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_3_multiply_num)

        self.pred_nn_4 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_3_multiply_num,
                                   self.mc.pred_dim * self.mc.pred_nn_4_multiply_num)
        self.pred_nn_4_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_4_multiply_num)

        self.pred_nn_5 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_4_multiply_num,
                                   self.mc.pred_dim * self.mc.pred_nn_5_multiply_num)
        self.pred_nn_5_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_5_multiply_num)

        self.pred_nn_6 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_5_multiply_num,
                                   1)

        # ++ ===========================================================================================================

        # 활성화 함수 및 드롭아웃 비율 정의
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.mc.dropout)

    def forward(self, x):
        # ++ ===========================================================================================================

        # FCNN 레이어 연결

        # 카테고리별 입력 추출
        all_input = x[:, self.idx_full_list]

        # concat → FCNN
        vec_concat = torch.cat([all_input], dim=1)  # [B, pred_dim]

        # 첫번째, 마지막 전 완전연결층은 dropout x -> 정보 보존 목적
        out = self.pred_nn_1(vec_concat)
        out = self.pred_nn_1_batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.pred_nn_2(out)
        out = self.pred_nn_2_batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.pred_nn_3(out)
        out = self.pred_nn_3_batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.pred_nn_4(out)
        out = self.pred_nn_4_batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.pred_nn_5(out)
        # out = self.pred_nn_5_batchnorm(out)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.pred_nn_6(out)
        out = self.sigmoid(out)
        return out
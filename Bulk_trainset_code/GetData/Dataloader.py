from Bulk_trainset_code.config import config

class GetData :
    def __init__(self, NEED_VAL = False) :
        import warnings
        from Bulk_trainset_code.Utils.GetData import Get
        warnings.filterwarnings('ignore')

        self.NEED_VAL = NEED_VAL
        self.dataloader = Get()
        self._loads_envinfo()
        self._get = {
            # 'env' : self.ENV_INFO,
            # 'slcm' : self.SLCM_INFO,
            'farms' : self.FRAMS_INFO,
            'asf_org': self.ASF_OCC_FARMS_INFO,
            'asf_farm_dist' : self.ASF_FARM_DIST,
            'asf_wild_dist': self.ASF_WILD_DIST,
            'farm_count_12km': self.FARM_COUNT_12KM,
            'visit' : self.VISIT_INFO,
            'manage_card': self.CARD_INFO,
            'wildboar_count_5km': self.WILDBOAR_COUNT_5KM,
            'habitat_dist': self.HABITAT_DIST,
            'farms_env_ratio': self.FARMS_ENV_RATIO,
            'farm_elevation': self.FARM_ELEVATION,
            'farms_habitat_possibility': self.FARMS_HABITAT_POSSIBILITY,
            'mother_avg_brd_had_co': self.MOTHER_AVG_BRD_HAD_CO
        }

    def __getitem__(self, item) :
        return self._get[item]

    def __len__(self) :
        return len(self.train)

    def _loads_envinfo(self) :
        self.FRAMS_INFO = self.dataloader.TransferDF(f''' select * FROM asf.nvqrs_final_step_20_23_0810 where std_dt between '{config.start_date}' and '{config.end_date}' ''')
        self.ASF_OCC_FARMS_INFO = self.dataloader.TransferDF(f''' select * from asf.asf_occ_farms_info ''')
        self.ASF_FARM_DIST = self.dataloader.TransferDF(f''' select * FROM asf.nearest_asf_farm_dist ''')
        self.ASF_WILD_DIST = self.dataloader.TransferDF(f''' select * FROM asf.nearest_wild_asf_dist ''')
        self.FARM_COUNT_12KM = self.dataloader.TransferDF(f''' select * FROM asf.pig_farm_count_12km ''')
        self.VISIT_INFO = self.dataloader.TransferDF(f''' select * FROM asf.tmp_farm_visit_count where std_dt < '{config.end_date}' ''')
        self.CARD_INFO = self.dataloader.TransferDF(f''' select * FROM asf.tn_aph_dsnfc_manage_frmhs_info ''')
        self.WILDBOAR_COUNT_5KM = self.dataloader.TransferDF(f''' select * from 검역본부2.농장반경5km_멧돼지수_재산출_2_20_23_new ''')
        self.HABITAT_DIST = self.dataloader.TransferDF(f''' select * from asf.nearest_habitat_dist ''')
        self.FARMS_ENV_RATIO = self.dataloader.TransferDF(f''' select * from asf.farms_env_ratio ''')
        self.FARM_ELEVATION = self.dataloader.TransferDF(f''' SELECT distinct on (farms_no) * FROM asf.farms_elevation ''')
        self.FARMS_HABITAT_POSSIBILITY = self.dataloader.TransferDF(f''' SELECT * FROM asf.farms_habitat_possibility ''')
        # self.MOTHER_AVG_BRD_HAD_CO = self.dataloader.TransferDF(f''' SELECT std_dt, farms_no, avg_mother_brd_had_co FROM asf.mother_pig_farms_avg_brd_had_co ''')
        self.MOTHER_AVG_BRD_HAD_CO = self.dataloader.TransferDF(f''' SELECT * FROM asf.nvqrs_final_step_mother_20_23 ''')

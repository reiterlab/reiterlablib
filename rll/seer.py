#!/usr/bin/python
"""Reading SEER data"""
import logging
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as stats

from rll.convert import *

__date__ = 'April 9, 2021'
__author__ = 'Johannes REITER'

# get logger
logger = logging.getLogger(__name__)


class Seer:

    c_id = 'PatientID'
    c_sex = 'Sex'
    c_size_mm = 'TumorSize_mm'  # tumor size
    c_size = 'TumorSize'  # tumor size
    c_vol = 'TumorVolume_cm3'  # tumor size
    c_cells = 'TumorVolumeCells'  # tumor size
    c_age = 'AgeDiagnosis'  # age at diagnosis [years]
    c_site = 'PrimarySite'  # location of primary tumor
    c_stage_seer = 'AJCCStage'
    c_stage_T = 'AJCCStageT'
    c_stage_N = 'AJCCStageN'
    c_stage_M = 'AJCCStageM'
    c_stage_met = 'StageEarlyLate'
    c_stage_simpl = 'Stage'
    c_lymmet = 'LymphaticMetastasis'
    c_dismet = 'DistantMetastasis'
    c_diag_month = 'DiagnosisMonth'
    c_diag_year = 'DiagnosisYear'
    c_survival = 'SurvivalMonths'
    c_survival_years = 'SurvivalYears'
    c_5year_surv = 'Survived_5years'  # boolean column encoding whether 5 years were survived (cancer-specific)
    c_dead_cancerdeath = 'Alive_CancerDeath'
    c_dead_otherdeath = 'Alive_OtherDeath'

    def __init__(self, seer_root_dir, start_year, end_year):
        """
        Initialize SEER data object
        :param seer_root_dir: root directory of the SEER data
        """

        self.root_dir = seer_root_dir
        self.start_year = start_year
        self.end_year = end_year

        self.incid_dir = os.path.join(self.root_dir, 'incidence', f'yr{start_year}_{end_year}.seer9')
        """
        BREAST.TXT    -  Breast
        COLRECT.TXT   -  Colon and Rectum
        DIGOTHR.TXT   -  Other Digestive
        FEMGEN.TXT    -  Female Genital
        LYMYLEUK.TXT  -  Lymphoma of All Sites and Leukemia
        MALEGEN.TXT   -  Male Genital
        RESPIR.TXT    -  Respiratory
        URINARY.TXT   -  Urinary
        OTHER.TXT     -  All Other Sites
        """

        self.crc_fp = os.path.join(self.incid_dir, 'COLRECT.TXT')
        self.resp_fp = os.path.join(self.incid_dir, 'RESPIR.TXT')
        self.dig_fp = os.path.join(self.incid_dir, 'DIGOTHR.TXT')
        self.brca_fp = os.path.join(self.incid_dir, 'BREAST.TXT')
        self.femalegen_fp = os.path.join(self.incid_dir, 'FEMGEN.TXT')
        self.malegen_fp = os.path.join(self.incid_dir, 'MALEGEN.TXT')

        self.site_filepaths = {'Lung': self.resp_fp, 'Colorectal': self.crc_fp, 'Pancreatic': self.dig_fp,
                               'Breast': self.brca_fp, 'Liver': self.dig_fp, 'Ovarian': self.femalegen_fp,
                               'Prostate': self.malegen_fp}
        self.fps = set(self.site_filepaths.values())

        self.fp_pop = os.path.join(self.root_dir, 'populations', 'white_black_other',
                                   f'yr{start_year}_{end_year}.seer9', 'singleages.txt')

        self.df_pop = None
        self.df_incid = None
        self.df_sur = None
        logger.info(f'SEER root directory {os.path.abspath(seer_root_dir)}.')

    def read_pop_data(self):
        """
        Read SEER population data
        # for file format see SEER_1973_2015_TEXTDATA/populations/popdic.html
        :return: dataframe with SEER population level data
        """
        colspecs = [
            (0, 4),  # year
            (4, 6),  # state postal abbreviation
            (6, 8),  # state FIPS code
            (8, 11),  # county FIPS code
            (11, 13),  # registry
            (13, 14),  # race
            (14, 15),  # origin
            (15, 16),  # sex
            (16, 18),  # age
            (18, 30)  # population
        ]
        col_names = ['year', 'state_post_abbr', 'state_fips_code', 'county_fips_code',
                     'registry', 'race', 'origin', 'sex', 'age', 'population']
        self.df_pop = pd.read_fwf(self.fp_pop, colspecs=colspecs, names=col_names)

        # since the incidence data are only reported for years 2005-2015, remove all other data from earlier years
        self.df_pop.drop(self.df_pop[self.df_pop['year'] < 2005].index, inplace=True)
        assert all(2005 <= self.df_pop['year'].unique()) and all(self.df_pop['year'].unique() <= self.end_year)
        self.df_pop['sex'].replace(1, 'male', inplace=True)
        self.df_pop['sex'].replace(2, 'female', inplace=True)

        return self.df_pop

    def read_incidence_data(self, sites=None, recency_cutoff=2005, surv_diag_years=None):
        """
        Read SEER incidence data
        :param sites: list of primary sites; if None, then all SEER data files are read (takes a long time)
        :param recency_cutoff: remove outdated data up to given year (default 2005)
        :param surv_diag_years: tuple of years to study survival times in that interval (default None; e.g. (2006, 2009)
        :return: dataframe with SEER incidence data
        """
        # ######################################################################
        # ######### READ INCIDENCE DATA #############
        # NOTE: positions need be shifted by one because it starts with zero
        d = defaultdict(list)

        # site codes according to International Classification of Diseases for Oncology, Third Edition (ICD-O-3)
        # for topography codes https://seer.cancer.gov/manuals/2018/appendixc.html
        # https://seer.cancer.gov/siterecode/icdo3_dwhoheme/index.html
        primary_sites = {'C180': 'Colorectal', 'C181': 'Colorectal', 'C182': 'Colorectal', 'C183': 'Colorectal',
                         'C184': 'Colorectal', 'C185': 'Colorectal', 'C186': 'Colorectal', 'C187': 'Colorectal',
                         'C188': 'Colorectal', 'C189': 'Colorectal', 'C260': 'Colorectal',  # colon and intestine
                         'C199': 'Colorectal', 'C209': 'Colorectal', 'C210': 'Colorectal', 'C211': 'Colorectal',
                         'C212': 'Colorectal', 'C218': 'Colorectal',  # rectum and anus
                         'C340': 'Lung', 'C341': 'Lung', 'C342': 'Lung', 'C343': 'Lung',
                         'C344': 'Lung', 'C345': 'Lung', 'C346': 'Lung', 'C347': 'Lung',
                         'C348': 'Lung', 'C349': 'Lung',  # lung and bronchus
                         'C500': 'Breast', 'C501': 'Breast', 'C502': 'Breast', 'C503': 'Breast', 'C504': 'Breast',
                         # breast
                         'C505': 'Breast', 'C506': 'Breast', 'C507': 'Breast', 'C508': 'Breast', 'C509': 'Breast',
                         'C250': 'Pancreatic', 'C251': 'Pancreatic', 'C252': 'Pancreatic', 'C253': 'Pancreatic',
                         'C254': 'Pancreatic', 'C255': 'Pancreatic', 'C256': 'Pancreatic', 'C257': 'Pancreatic',
                         'C258': 'Pancreatic', 'C259': 'Pancreatic',  # pancreas
                         'C220': 'Liver',  # liver
                         }

        if sites is None:
            file_paths = self.fps
        else:
            file_paths = set([self.site_filepaths[site] for site in sites])

        for seer_fp in file_paths:
            with open(seer_fp, 'r') as f_seer:
                for line in f_seer:

                    #  if a list of primary sites is given only keep those entries with any of the given primary sites
                    if sites is not None:
                        if line[42:46] not in primary_sites.keys() or primary_sites[line[42:46]] not in sites:
                            continue

                    if line[42:46] in primary_sites.keys():
                        d[Seer.c_site].append(primary_sites[line[42:46]])
                    else:
                        d[Seer.c_site].append(line[42:46])

                    # NOTE: positions need be shifted by one because it starts with zero
                    d[Seer.c_id].append(line[0:8])
                    # sex/gender
                    d[Seer.c_sex].append('male' if line[23] == '1' else 'female')
                    # age at diagnosis in years
                    d[Seer.c_age].append(line[24:27])
                    # month and year of diagnosis
                    d[Seer.c_diag_month].append(line[36:38])
                    d[Seer.c_diag_year].append(line[38:42])

                    # tumor size: Information on tumor size. Available for 2004-2015 diagnosis years.
                    d[Seer.c_size_mm].append(line[95:98])
                    # see stage formatting here: https://seer.cancer.gov/seerstat/variables/seer/ajcc-stage/3rd.html
                    d[Seer.c_stage_T].append(line[127:129])
                    d[Seer.c_stage_N].append(line[129:131])
                    d[Seer.c_stage_M].append(line[131:133])  # NAACCR Item #: 2850
                    #             d[c_stage].append(line[236:238])  # outdated (given only up to 2003)
                    d[Seer.c_stage_seer].append(line[133:135])  # outdated (given only up to 2003)

                    # survival
                    # survival months flag (305) needs to be 1 for complete date information
                    if line[304] == '1':
                        # survival months position 301-304
                        d[Seer.c_survival].append(line[300:304])
                    else:
                        d[Seer.c_survival].append(np.nan)

                    # SEER cause-specific death classification 272
                    # 1 = dead due to cancer; 0 = alive or dead of other cause
                    d[Seer.c_dead_cancerdeath].append(line[271])

                    # SEER other cause of death classification 273
                    # 1 = dead due to other casue; 0 = alive or dead due to cancer
                    d[Seer.c_dead_otherdeath].append(line[272])

        df_incid = pd.DataFrame(data=d)

        df_incid.replace('   ', np.nan, inplace=True)
        df_incid[Seer.c_size_mm].replace('999', np.nan, inplace=True)  # size unknown
        df_incid[Seer.c_size_mm].replace('888', np.nan, inplace=True)  # not applicable
        df_incid[Seer.c_size_mm].replace('000', np.nan, inplace=True)  # no primary tumor was found
        df_incid[Seer.c_size_mm].replace('991', np.nan, inplace=True)  # described as less than 1 cm
        df_incid[Seer.c_size_mm].replace('992', np.nan, inplace=True)  # described as less than 2 cm
        df_incid[Seer.c_size_mm].replace('993', np.nan, inplace=True)  # described as less than 3 cm
        df_incid[Seer.c_size_mm].replace('994', np.nan, inplace=True)  # described as less than 4 cm
        df_incid[Seer.c_size_mm].replace('995', np.nan, inplace=True)  # described as less than 5 cm
        for code in ['996', '997', '998']:
            df_incid[Seer.c_size_mm].replace(code, np.nan, inplace=True)  # site-specific code needed
        df_incid[Seer.c_size_mm] = df_incid[Seer.c_size_mm].astype(np.float64)
        df_incid[Seer.c_size_mm].replace(0.0, np.nan, inplace=True)
        # convert size from mm to cm
        df_incid[Seer.c_size] = df_incid.apply(lambda row: row[Seer.c_size_mm] / 10.0, axis=1)
        # calculate tumor volumes
        df_incid[Seer.c_vol] = df_incid.apply(lambda row: diameter_volume(row[Seer.c_size]), axis=1)
        # calculate number of cells
        df_incid[Seer.c_cells] = df_incid.apply(lambda row: diameter_cells(row[Seer.c_size]), axis=1)

        df_incid[Seer.c_age] = df_incid[Seer.c_age].astype(np.float64)
        df_incid[Seer.c_diag_year] = df_incid[Seer.c_diag_year].astype(np.float64)
        # 00	Tis
        # 88	Recode scheme not yet available
        # 90	Unstaged
        # 98	Not applicable
        # 99	Error condition
        excluding_T_stages = set(['00', '99', '01', '05', '88'])

        # if lymph nodes could not be assessed assign null
        df_incid[Seer.c_lymmet] = df_incid.apply(
            lambda row: pd.NA if row[Seer.c_stage_N] == '99' else (False if row[Seer.c_stage_N] == '00' else True),
            axis=1)
        # if distant metastases could not be assessed assign null
        df_incid[Seer.c_dismet] = df_incid.apply(
            lambda row: pd.NA if row[Seer.c_stage_M] == '99' else (False if row[Seer.c_stage_M] == '00' else True),
            axis=1)

        # Simplify cancer stages
        df_incid[Seer.c_stage_seer].replace('  ', np.nan, inplace=True)
        df_incid = df_incid.astype({Seer.c_stage_seer: np.float64})

        def simplify_seer_stage(agcc_stage_grp_code):
            if agcc_stage_grp_code <= 2:
                return 0
            elif agcc_stage_grp_code < 30:
                return 1
            elif agcc_stage_grp_code < 50:
                return 2
            elif agcc_stage_grp_code < 70:
                return 3
            elif agcc_stage_grp_code <= 74:
                return 4
            else:
                return np.nan

        def simplify_seer_stage_met(ajcc_stage):
            if ajcc_stage == 1 or ajcc_stage == 2:
                return 'early'
            elif ajcc_stage == 3 or ajcc_stage == 4:
                return 'late'
            else:
                return np.nan

        df_incid[Seer.c_stage_simpl] = df_incid.apply(lambda row: simplify_seer_stage(row[Seer.c_stage_seer]), axis=1)
        df_incid[Seer.c_stage_met] = df_incid.apply(lambda row: simplify_seer_stage_met(row[Seer.c_stage_simpl]),
                                                    axis=1)

        # survival months
        df_incid[Seer.c_survival].replace('9999', np.nan, inplace=True)
        df_incid[Seer.c_survival] = df_incid[Seer.c_survival].astype(np.float64)
        df_incid[Seer.c_survival_years] = df_incid[Seer.c_survival] / 12

        # boolean 5 year cancer-specific survival
        df_incid[Seer.c_5year_surv] = df_incid.apply(lambda row: np.nan if np.isnan(row[Seer.c_survival]) else
                                                     (True if row[Seer.c_survival] >= 60 else False), axis=1)

        # death classification
        df_incid[Seer.c_dead_otherdeath].replace('9', np.nan, inplace=True)
        df_incid[Seer.c_dead_cancerdeath].replace('9', np.nan, inplace=True)

        if recency_cutoff is not None:
            self.df_incid = df_incid.drop(
                df_incid[(df_incid[Seer.c_diag_year] < recency_cutoff)
                         | (df_incid[Seer.c_stage_T].isin(excluding_T_stages))].index)
            logger.info(f'Remaining entries diagnosed {recency_cutoff} or later: {len(self.df_incid)}')
            # data_df = df.drop(df[(df[Seer.c_diag_year] < 2005) | (df.AJCCStageT.isin(excluding_stages))].index)
            # print(f'Remaining entries with a recorded tumor size: {len(data_df[data_df.TumorSize.notnull()])}')
        else:
            self.df_incid = df_incid

        # sort dataframe by diagnosis date such that only the first cancer is kept
        self.df_incid.sort_values(by=[Seer.c_diag_year, Seer.c_diag_month], inplace=True)
        self.df_incid.drop_duplicates(subset=[Seer.c_id], keep='first', inplace=True)
        logger.info(f'Remaining entries after removing second diagnosed cancers: {len(self.df_incid)}')

        if surv_diag_years is not None:
            # filter for cancers that were diagnosed at least 6 years before the cutoff to get sufficient follow-up
            df_sur = df_incid.drop(df_incid[(df_incid[Seer.c_diag_year] > surv_diag_years[1])
                                            | (df_incid[Seer.c_diag_year] < surv_diag_years[0])
                                            | (df_incid[Seer.c_stage_T].isin(excluding_T_stages))].index)

            # sort dataframe by diagnosis date such that only the first cancer is kept
            df_sur.sort_values(by=[Seer.c_diag_year, Seer.c_diag_month], inplace=True)
            df_sur.drop_duplicates(subset=[Seer.c_id], keep='first', inplace=True)
            logger.info('Remaining entries after removing second diagnosed cancers: {}'.format(len(df_sur)))
            self.df_sur = df_sur

        for site in self.df_incid[Seer.c_site].unique():
            logger.info(f'Read {self.df_incid[self.df_incid[Seer.c_site] == site].shape[0]} cases with {site} cancer.')

        return self.df_incid, self.df_sur

    def print_site_summary(self, site):

        site_filt = self.df_incid[Seer.c_site] == site
        cs_death_filt = self.df_sur[Seer.c_dead_otherdeath] == '0'

        print('{} {} cases: detection size [cm]: '.format(
            self.df_incid[site_filt][Seer.c_size].count(), site)
              + 'mean {:.2f} ({:.3f} cm3), median {:.2f} ({:.3f} cm3), IQR: {}-{}'.format(
            np.nanmean(self.df_incid[site_filt][Seer.c_size]),
            np.nanmean(self.df_incid[site_filt][Seer.c_vol]),
            np.nanmedian(self.df_incid[site_filt][Seer.c_size]),
            np.nanmedian(self.df_incid[site_filt][Seer.c_vol]),
            np.nanpercentile(self.df_incid[site_filt][Seer.c_size], 25),
            np.nanpercentile(self.df_incid[site_filt][Seer.c_size], 75)))

        print('{} detection age mean: {:.2f}, median {}, IQR: {}-{} years'.format(
            site, np.nanmean(self.df_incid[site_filt][Seer.c_age]),
            np.nanmedian(self.df_incid[site_filt][Seer.c_age]),
            np.nanpercentile(self.df_incid[site_filt][Seer.c_age], 25),
            np.nanpercentile(self.df_incid[site_filt][Seer.c_age], 75)))

        if self.df_sur is not None:
            print('{} survival mean: {:.2f}, median {}, IQR: {}-{} months'.format(
                site, np.nanmean(self.df_sur[(self.df_sur[Seer.c_site] == site)][Seer.c_survival]),
                np.nanmedian(self.df_sur[self.df_sur[Seer.c_site] == site][Seer.c_survival]),
                np.nanpercentile(self.df_sur[self.df_sur[Seer.c_site] == site][Seer.c_survival], 25),
                np.nanpercentile(self.df_sur[self.df_sur[Seer.c_site] == site][Seer.c_survival], 75)))

            print('{} cancer-specific survival mean: {:.2f}, median {}, IQR: {}-{} months'.format(
                site, np.nanmean(self.df_sur[(self.df_sur[Seer.c_site] == site) & cs_death_filt][Seer.c_survival]),
                np.nanmedian(self.df_sur[(self.df_sur[Seer.c_site] == site) & cs_death_filt][Seer.c_survival]),
                np.nanpercentile(self.df_sur[(self.df_sur[Seer.c_site] == site) & cs_death_filt][Seer.c_survival], 25),
                np.nanpercentile(self.df_sur[(self.df_sur[Seer.c_site] == site) & cs_death_filt][Seer.c_survival], 75)))

        # check correlation between tumor size and age at diagnosis
        site_df = self.df_incid[site_filt & np.isfinite(self.df_incid[site_filt][Seer.c_age])
                                & np.isfinite(self.df_incid[site_filt][Seer.c_size])]
        spearman = stats.spearmanr(site_df[site_df[Seer.c_site] == site][Seer.c_age],
                                   site_df[site_df[Seer.c_site] == site][Seer.c_size])
        print(f'{site} Spearman\'s rho correlation between age and size at diagnosis: {spearman[0]:.3f} '
              + f'(p={spearman[1]:.3e})')

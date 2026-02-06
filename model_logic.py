# -*- coding: utf-8 -*-
"""
model_logic.py: Core Machine Learning Engine
Refactored for Docker/FastAPI compatibility.
"""

import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

class Config:
    # Docker-friendly paths using Environment Variables
    # Defaults to internal Docker paths if variables aren't set
    BASE_DIRECTORY = os.getenv('DATA_PATH', '/app/data/csvs/')
    GEODATA_FILENAME = os.getenv('GEO_PATH', '/app/data/tanzania_council_geodata.csv')
    
    EXPECTED_COLS = {
        'Year': ['Year', 'YEAR', 'Academic Year'],
        'Region': ['Region', 'REGION', 'REGON'],
        'Council': ['Council', 'COUNCIL', 'DISTRICT', 'LGA NAME']
    }
    
    EXCLUDE_KEYWORDS = ['Primary', 'Textbooks', 'Population', 'COBET', 'Vocational']

class DataHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.dataframes = {}

    def load_data(self):
        try:
            if not os.path.exists(self.base_dir):
                print(f"Directory not found: {self.base_dir}")
                return

            all_files_in_dir = os.listdir(self.base_dir)
            all_files = [f for f in all_files_in_dir if f.endswith('.csv')]
            
            filtered_files = []
            for file_name in all_files:
                if not any(keyword.lower() in file_name.lower() for keyword in Config.EXCLUDE_KEYWORDS):
                    filtered_files.append(file_name)
            
            for file_name in filtered_files:
                file_path = os.path.join(self.base_dir, file_name)
                df_name = file_name.replace('.csv', '')
                try:
                    self.dataframes[df_name] = pd.read_csv(file_path)
                    print(f"Loaded {df_name}.")
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
        except FileNotFoundError:
            print(f"Directory not found: {self.base_dir}")

    def clean_data(self):
        for df_name, df in self.dataframes.items():
            for col in df.select_dtypes(include='object').columns:
                if df[col].astype(str).str.contains(',').any() or df[col].astype(str).str.fullmatch(r'\d+\.?\d*').any():
                    cleaned_col = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
                    converted_col = pd.to_numeric(cleaned_col, errors='coerce')
                    if converted_col.notna().sum() > 0:
                        df[col] = converted_col

            unnamed_cols = [col for col in df.columns if re.match(r'Unnamed: \d+', str(col))]
            cols_to_drop = [col for col in unnamed_cols if (df[col].isnull().sum() / len(df) * 100) > 90]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)

    def merge_lga_status(self):
        if 'LGAs Urban and Rural Status' not in self.dataframes: return

        df_lga = self.dataframes['LGAs Urban and Rural Status'].copy()
        if 'Remarks' in df_lga.columns: df_lga.drop(columns=['Remarks'], inplace=True)
        
        df_lga.rename(columns={'Region': 'Region', 'Council': 'Council', 'Classification': 'LGA_Status'}, inplace=True)
        df_lga['Region'] = df_lga['Region'].str.upper().str.strip()
        df_lga['Council'] = df_lga['Council'].str.upper().str.strip()

        for df_name, df in self.dataframes.items():
            if df_name == 'LGAs Urban and Rural Status': continue
            
            actual_region = next((n for n in Config.EXPECTED_COLS['Region'] if n in df.columns), None)
            actual_council = next((n for n in Config.EXPECTED_COLS['Council'] if n in df.columns), None)

            if actual_region and actual_council:
                df[actual_region] = df[actual_region].astype(str).str.upper().str.strip()
                df[actual_council] = df[actual_council].astype(str).str.upper().str.strip()
                try:
                    merged_df = pd.merge(df, df_lga, 
                                         left_on=[actual_region, actual_council], 
                                         right_on=['Region', 'Council'], 
                                         how='left', suffixes=('', '_LGA'))
                    if 'Region_LGA' in merged_df.columns: merged_df.drop(columns=['Region_LGA'], inplace=True)
                    if 'Council_LGA' in merged_df.columns: merged_df.drop(columns=['Council_LGA'], inplace=True)
                    self.dataframes[df_name] = merged_df
                except Exception: pass

    def drop_null_locations(self):
        for df_name, df in self.dataframes.items():
            actual_region = next((n for n in Config.EXPECTED_COLS['Region'] if n in df.columns), None)
            actual_council = next((n for n in Config.EXPECTED_COLS['Council'] if n in df.columns), None)
            if actual_region and actual_council:
                df.dropna(subset=[actual_region, actual_council], inplace=True)

    def get_dataframe(self, name):
        return self.dataframes.get(name)

class GeoProcessor:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.geo_data = None

    def process_geodata(self):
        if os.path.exists(Config.GEODATA_FILENAME):
            print(f"Loading geodata from {Config.GEODATA_FILENAME}")
            self.geo_data = pd.read_csv(Config.GEODATA_FILENAME)
        
            self.geo_data['Latitude'] = pd.to_numeric(self.geo_data['Latitude'], errors='coerce')
            self.geo_data['Longitude'] = pd.to_numeric(self.geo_data['Longitude'], errors='coerce')
            self.geo_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

            X = self.geo_data[['Latitude', 'Longitude']]
            if len(X) > 0:
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                self.geo_data['Geo_Cluster'] = kmeans.fit_predict(X)

            self._merge_geodata_to_dataframes()

    def _merge_geodata_to_dataframes(self):
        if self.geo_data is None: return
        for df_name, df in self.data_handler.dataframes.items():
            actual_region = next((n for n in Config.EXPECTED_COLS['Region'] if n in df.columns), None)
            actual_council = next((n for n in Config.EXPECTED_COLS['Council'] if n in df.columns), None)

            if actual_region and actual_council:
                df[actual_region] = df[actual_region].astype(str).str.upper().str.strip()
                df[actual_council] = df[actual_council].astype(str).str.upper().str.strip()
                try:
                    merged_df = pd.merge(df, self.geo_data[['Region', 'Council', 'Geo_Cluster']],
                                         left_on=[actual_region, actual_council],
                                         right_on=['Region', 'Council'],
                                         how='left', suffixes=('', '_geo'))
                    if 'Region_geo' in merged_df.columns: merged_df.drop(columns=['Region_geo'], inplace=True)
                    if 'Council_geo' in merged_df.columns: merged_df.drop(columns=['Council_geo'], inplace=True)
                    self.data_handler.dataframes[df_name] = merged_df
                except Exception: pass

class SubjectModeler:
    def __init__(self, data_handler):
        self.dh = data_handler
        self.govt_ratios = None

    @staticmethod
    def standard_cols(df):
        if df is None: return pd.DataFrame()
        df = df.copy()
        df.columns = [str(c).strip().upper() for c in df.columns]
        rename_map = {
            'YEAR': 'YEAR', 'REGION': 'REGION', 'COUNCIL': 'COUNCIL',
            'LGA_STATUS': 'LGA_STATUS', 'GEO_CLUSTER': 'GEO_CLUSTER'
        }
        df.rename(columns=rename_map, inplace=True)
        return df.loc[:, ~df.columns.duplicated()]

    def calculate_govt_ratios(self):
        print("\nCalculating Government vs All Ratios from aggregate files...")
        df_all = self.dh.get_dataframe("Secondary_enrol_all_2016_2025")
        df_gov = self.dh.get_dataframe("Secondary_enrol_Gov_2016_2025")

        if df_all is None or df_gov is None:
            print("Warning: Aggregate enrollment files not found. Ratios cannot be calculated.")
            return

        df_all = self.standard_cols(df_all)
        df_gov = self.standard_cols(df_gov)

        keys = ['YEAR', 'REGION', 'COUNCIL']
        all_cols = set(df_all.columns)
        gov_cols = set(df_gov.columns)
        common_cols = all_cols.intersection(gov_cols)
        
        form_candidates = [c for c in common_cols if c not in keys and 'TOTAL' not in c]
        
        def melt_agg(df, val_name):
            return df.melt(id_vars=[k for k in keys if k in df.columns], 
                           value_vars=[c for c in form_candidates if c in df.columns],
                           var_name='RAW_FORM', value_name=val_name)

        long_all = melt_agg(df_all, 'ENROL_ALL')
        long_gov = melt_agg(df_gov, 'ENROL_GOV')

        # Clean numeric data
        long_all['ENROL_ALL'] = pd.to_numeric(long_all['ENROL_ALL'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        long_gov['ENROL_GOV'] = pd.to_numeric(long_gov['ENROL_GOV'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

        def extract_form(s):
            match = re.search(r'(\d+)', str(s))
            if match: return int(match.group(1))
            romans = {'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'VI':6}
            for r, n in romans.items():
                if str(s).endswith(f" {r}") or str(s) == r: return n
            return -1

        long_all['FORM_NUM'] = long_all['RAW_FORM'].apply(extract_form)
        long_gov['FORM_NUM'] = long_gov['RAW_FORM'].apply(extract_form)

        merged = pd.merge(long_all, long_gov, on=keys + ['FORM_NUM'], how='inner')
        
        merged['GOVT_RATIO'] = merged['ENROL_GOV'] / (merged['ENROL_ALL'] + 1e-9)
        merged['GOVT_RATIO'] = merged['GOVT_RATIO'].clip(0, 1)

        self.govt_ratios = merged[keys + ['FORM_NUM', 'GOVT_RATIO']]
        print(f" > Ratios calculated for {len(self.govt_ratios)} Year-Region-Council-Form combinations.")

    def prepare_data(self):
        df_subject = self.dh.get_dataframe("Secondary_students_per_subject")
        if df_subject is None:
            print("Error: Secondary_students_per_subject not found.")
            return None

        aux_files = {
            'tables': "Data-Secondary Tables and chairs 2016-2025",
            'drops': "Dropout-Secondary  2017-2024",
            'reentry': "Secondary-Re_entry",
            'disability': "Secondary - DISABALITY 2020-2025",
            'ict': "Combined_Secondary_ICT_All_G_NG",
            'elec': "Combined_Secondary_Electricity_All_G_NG",
            'labs': "Combined_Secondary_Laboratories_All_G_NG"
        }
        aux_dfs = {k: self.standard_cols(self.dh.get_dataframe(v)) for k,v in aux_files.items()}

        enroll = self.standard_cols(df_subject)
        id_vars = [c for c in ['YEAR', 'REGION', 'COUNCIL'] if c in enroll.columns]
        subject_cols = [c for c in enroll.columns if 'FORM ' in c and ' - ' in c]
        
        long_df = enroll.melt(id_vars=id_vars, value_vars=subject_cols, var_name='RAW', value_name='ENROLLMENT')
        
        long_df['ENROLLMENT'] = pd.to_numeric(long_df['ENROLLMENT'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        long_df['FORM_NUM'] = long_df['RAW'].str.extract(r'FORM (\d)').astype(int)
        long_df['SUBJECT'] = long_df['RAW'].str.split(' - ').str[1].str.strip()
        long_df.drop(columns=['RAW'], inplace=True)

        keys = ['YEAR', 'REGION', 'COUNCIL']

        def merge_aux(main, aux, cols, fill_val=0):
            if aux.empty: 
                for c in cols: main[c] = fill_val
                return main
            valid_cols = [c for c in cols if c in aux.columns]
            if not valid_cols:
                for c in cols: main[c] = fill_val
                return main
            
            if len(aux) > len(aux[keys].drop_duplicates()):
                aux = aux.groupby(keys)[valid_cols].sum().reset_index()

            temp = pd.merge(main, aux[keys + valid_cols], on=keys, how='left')
            for c in cols: 
                if c not in temp.columns: temp[c] = fill_val
                else: 
                    temp[c] = pd.to_numeric(temp[c].astype(str).str.replace(',', ''), errors='coerce').fillna(fill_val)
            return temp

        long_df = merge_aux(long_df, aux_dfs['tables'], ['AVAILABLE_TABLES'])
        
        lab_cols = [c for c in aux_dfs['labs'].columns if 'LABORATORY' in c]
        if lab_cols:
            for lc in lab_cols: aux_dfs['labs'][lc] = pd.to_numeric(aux_dfs['labs'][lc], errors='coerce').fillna(0)
            aux_dfs['labs']['TOTAL_LABS'] = aux_dfs['labs'][lab_cols].sum(axis=1)
            long_df = merge_aux(long_df, aux_dfs['labs'], ['TOTAL_LABS'])
        else: long_df['TOTAL_LABS'] = 0

        re_cols = [c for c in aux_dfs['reentry'].columns if 'RE-ENROLLED' in c]
        if re_cols:
            for rc in re_cols: aux_dfs['reentry'][rc] = pd.to_numeric(aux_dfs['reentry'][rc], errors='coerce').fillna(0)
            aux_dfs['reentry']['TOTAL_REENTRY'] = aux_dfs['reentry'][re_cols].sum(axis=1)
            long_df = merge_aux(long_df, aux_dfs['reentry'], ['TOTAL_REENTRY'])
        else: long_df['TOTAL_REENTRY'] = 0

        dis_cols = [c for c in aux_dfs['disability'].columns if c in ['BLIND', 'LOW VISION']]
        if dis_cols:
            for dc in dis_cols: aux_dfs['disability'][dc] = pd.to_numeric(aux_dfs['disability'][dc], errors='coerce').fillna(0)
            aux_dfs['disability']['TOTAL_DISABLED'] = aux_dfs['disability'][dis_cols].sum(axis=1)
            long_df = merge_aux(long_df, aux_dfs['disability'], ['TOTAL_DISABLED'])
        else: long_df['TOTAL_DISABLED'] = 0

        ict_cols = [c for c in aux_dfs['ict'].columns if 'COMPUTERS' in c]
        if ict_cols:
            for ic in ict_cols: aux_dfs['ict'][ic] = pd.to_numeric(aux_dfs['ict'][ic], errors='coerce').fillna(0)
            aux_dfs['ict']['TOTAL_COMPUTERS'] = aux_dfs['ict'][ict_cols].sum(axis=1)
            long_df = merge_aux(long_df, aux_dfs['ict'], ['TOTAL_COMPUTERS'])
        else: long_df['TOTAL_COMPUTERS'] = 0

        elec_raw_col = 'NATIONAL GRID (TANESCO) %'
        if elec_raw_col in aux_dfs['elec'].columns:
            aux_dfs['elec'].rename(columns={elec_raw_col: 'ELEC_GRID_PCT'}, inplace=True)
            long_df = merge_aux(long_df, aux_dfs['elec'], ['ELEC_GRID_PCT'])
        else: long_df['ELEC_GRID_PCT'] = 0

        d_cols = ['TRUANCY', 'PREGNANCY', 'INDISCIPLINE']
        long_df = merge_aux(long_df, aux_dfs['drops'], d_cols)

        return long_df.sort_values(['REGION', 'COUNCIL', 'SUBJECT', 'FORM_NUM', 'YEAR'])

    def engineer_features(self, df):
        g = df.groupby(['REGION', 'COUNCIL', 'SUBJECT', 'FORM_NUM'])
        df['LAG_1'] = g['ENROLLMENT'].shift(1)
        df['LAG_2'] = g['ENROLLMENT'].shift(2)
        df['YOY_GROWTH'] = (df['ENROLLMENT'] - df['LAG_1']) / (df['LAG_1'] + 1e-5)

        df['PREV_YEAR'] = df['YEAR'] - 1
        df['PREV_FORM'] = df['FORM_NUM'] - 1

        df['LOOKUP_KEY'] = (df['YEAR'].astype(str) + '_' + df['REGION'] + '_' +
                            df['COUNCIL'] + '_' + df['SUBJECT'] + '_' + df['FORM_NUM'].astype(str))
        
        lookup = df.groupby('LOOKUP_KEY')['ENROLLMENT'].sum().to_dict()

        df['SEARCH_KEY'] = (df['PREV_YEAR'].astype(str) + '_' + df['REGION'] + '_' +
                            df['COUNCIL'] + '_' + df['SUBJECT'] + '_' + df['PREV_FORM'].astype(str))
        
        df['COHORT_LAG'] = df['SEARCH_KEY'].map(lookup).fillna(-1)
        
        df.drop(columns=['PREV_YEAR', 'PREV_FORM', 'LOOKUP_KEY', 'SEARCH_KEY'], inplace=True)
        df['IS_ELECTION_YEAR'] = df['YEAR'].isin([2015, 2020, 2025, 2030, 2035]).astype(int)
        
        return df.fillna(-1)

    def calculate_metrics(self, y_true, y_pred, model_name, year_label):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
            
        accuracy = 100 - mape if not np.isnan(mape) else 0
        
        print(f" > {model_name} [{year_label}]: RMSE={rmse:,.0f} | MAE={mae:,.0f} | MAPE={mape:.2f}% | Acc={accuracy:.2f}%")
        return rmse

    def run_modeling(self, df):
        le_reg = LabelEncoder()
        le_cou = LabelEncoder()
        le_sub = LabelEncoder()

        df['REGION_ENC'] = le_reg.fit_transform(df['REGION'].astype(str))
        df['COUNCIL_ENC'] = le_cou.fit_transform(df['COUNCIL'].astype(str))
        df['SUBJECT_ENC'] = le_sub.fit_transform(df['SUBJECT'].astype(str))

        feats = [
            'YEAR', 'REGION_ENC', 'COUNCIL_ENC', 'SUBJECT_ENC', 'FORM_NUM',
            'AVAILABLE_TABLES', 'TOTAL_REENTRY', 'TOTAL_DISABLED',
            'TOTAL_COMPUTERS', 'ELEC_GRID_PCT', 'TOTAL_LABS',
            'TRUANCY', 'PREGNANCY', 'INDISCIPLINE',
            'LAG_1', 'LAG_2', 'YOY_GROWTH', 'COHORT_LAG', 'IS_ELECTION_YEAR'
        ]

        available_feats = [f for f in feats if f in df.columns]
        
        # 1. TRAIN on Data BEFORE 2025
        print(f"\nTraining Models on Data Before 2025 (Years < 2025)...")
        train_df = df[df['YEAR'] < 2025].copy()
        X_train = train_df[available_feats]
        y_train = train_df['ENROLLMENT']

        models = {
            "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=9, learning_rate=0.05, n_jobs=-1),
            "LightGBM": lgb.LGBMRegressor(n_estimators=200, num_leaves=50, min_child_samples=10, learning_rate=0.1, verbose=-1),
            "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42),
        }

        trained_models = {}
        for name, model in models.items():
            print(f" > Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model

        # 2. SELECTION: Evaluate on 2025 Data as Validation
        print("\nEvaluating Models on 2025 Data for Ensemble Selection...")
        val_df = df[df['YEAR'] == 2025].copy()
        
        selected_models = []
        if val_df.empty:
            print("Warning: No 2025 data found for validation. Using all models.")
            selected_models = list(trained_models.keys())
        else:
            X_val = val_df[available_feats]
            y_val = val_df['ENROLLMENT']
            
            model_metrics = {}
            for name, model in trained_models.items():
                pred = model.predict(X_val)
                rmse = self.calculate_metrics(y_val, pred, name, "2025")
                model_metrics[name] = rmse
            
            best_model_name = min(model_metrics, key=model_metrics.get)
            best_rmse = model_metrics[best_model_name]
            threshold = best_rmse * 1.20 # Within 20% error
            
            selected_models = [name for name, rmse in model_metrics.items() if rmse <= threshold]
            print(f" > Best Model: {best_model_name} (RMSE: {best_rmse:,.0f})")
            print(f" > Selected Ensemble Models (within 20%): {selected_models}")

        # 3. FORECAST for 2026-2030
        print("\nGenerating Forecasts for 2026-2030 using Ensemble...")
        
        forecast_years = [2026, 2027, 2028, 2029, 2030]
        results = []
        
        current_data = df.copy()

        for year in forecast_years:
            base_structure = current_data[current_data['YEAR'] == 2025][['REGION', 'COUNCIL', 'SUBJECT', 'FORM_NUM', 'REGION_ENC', 'COUNCIL_ENC', 'SUBJECT_ENC']].copy()
            
            if base_structure.empty:
                print(f"Error: No base structure found for generating {year} forecast.")
                break

            base_structure['YEAR'] = year
            base_structure['IS_ELECTION_YEAR'] = 1 if year in [2025, 2030, 2035] else 0
            
            # Update LAGs
            lag1_source = current_data[current_data['YEAR'] == year - 1].set_index(['REGION', 'COUNCIL', 'SUBJECT', 'FORM_NUM'])['ENROLLMENT'].to_dict()
            base_structure['LAG_1'] = base_structure.set_index(['REGION', 'COUNCIL', 'SUBJECT', 'FORM_NUM']).index.map(lag1_source).fillna(0)
            
            lag2_source = current_data[current_data['YEAR'] == year - 2].set_index(['REGION', 'COUNCIL', 'SUBJECT', 'FORM_NUM'])['ENROLLMENT'].to_dict()
            base_structure['LAG_2'] = base_structure.set_index(['REGION', 'COUNCIL', 'SUBJECT', 'FORM_NUM']).index.map(lag2_source).fillna(0)
            
            base_structure['YOY_GROWTH'] = (base_structure['LAG_1'] - base_structure['LAG_2']) / (base_structure['LAG_2'] + 1e-5)
            
            for feat in available_feats:
                if feat not in base_structure.columns:
                    base_structure[feat] = 0 

            X_test = base_structure[available_feats]
            
            ensemble_preds = []
            for name in selected_models:
                p = trained_models[name].predict(X_test)
                ensemble_preds.append(p)
            
            avg_pred = np.mean(ensemble_preds, axis=0)
            avg_pred = np.maximum(avg_pred, 0)
            
            base_structure['ENROLLMENT'] = avg_pred
            current_data = pd.concat([current_data, base_structure])

            temp = base_structure.copy()
            temp['ENROLLMENT_All'] = avg_pred
            
            if self.govt_ratios is not None:
                temp = pd.merge(temp, self.govt_ratios, 
                                on=['YEAR', 'REGION', 'COUNCIL', 'FORM_NUM'], 
                                how='left')
                
                avg_ratio = self.govt_ratios['GOVT_RATIO'].mean() if not self.govt_ratios.empty else 0.7
                temp['GOVT_RATIO'] = temp['GOVT_RATIO'].fillna(avg_ratio)
                
                temp['ENROLLMENT_Government'] = (temp['ENROLLMENT_All'] * temp['GOVT_RATIO']).round(0).astype(int)
            else:
                temp['ENROLLMENT_Government'] = 0

            temp['ENROLLMENT_All'] = temp['ENROLLMENT_All'].round(0).astype(int)
            
            results.append(temp)

        if results:
            final_df = pd.concat(results)
            # Return specific columns needed for the API/Database
            return final_df[['YEAR', 'REGION', 'COUNCIL', 'FORM_NUM', 'SUBJECT', 'ENROLLMENT_Government', 'ENROLLMENT_All']]
        
        return None

class Pipeline:
    def __init__(self):
        self.data_handler = DataHandler(Config.BASE_DIRECTORY)
        self.geo_processor = GeoProcessor(self.data_handler)
        self.modeler = SubjectModeler(self.data_handler)

    def run(self):
        print("\n=== Starting Modeling Pipeline ===")
        self.data_handler.load_data()
        self.data_handler.clean_data()
        self.data_handler.merge_lga_status()
        self.data_handler.drop_null_locations()
        self.geo_processor.process_geodata()

        self.modeler.calculate_govt_ratios()
        processed_df = self.modeler.prepare_data()
        
        if processed_df is not None:
            engineered_df = self.modeler.engineer_features(processed_df)
            return self.modeler.run_modeling(engineered_df)
        return None
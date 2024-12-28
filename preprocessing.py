from imputers import Imputers
import pandas as pd
import numpy as np
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import logging

def impute_pciat_values(train_df, max_missing=5, n_neighbors=5, imputer_type='knn', 
                        strategy='mean', max_iter=10, random_state=42): # TRY DIFFERENT IMPUTERS
    """
    Impute missing PCIAT values for rows that have at most max_missing missing values.
    
    Args:
        train_df (pd.DataFrame): Input dataframe containing PCIAT columns
        max_missing (int): Maximum number of missing values allowed for a row to be imputed
        
    Returns:
        pd.DataFrame: DataFrame with imputed PCIAT values
    """
    # Get PCIAT columns excluding PCIAT-Season and PCIAT-PCIAT_Total
    pciat_cols = [col for col in train_df.columns if col.startswith('PCIAT') 
                  and col != 'PCIAT-Season' 
                  and col != 'PCIAT-PCIAT_Total']
    
    # Count missing values per row
    missing_counts = train_df[pciat_cols].isnull().sum(axis=1)
    
    # Get rows with max_missing or fewer missing PCIAT values
    rows_to_impute = missing_counts <= max_missing
    print(f"Number of rows with {max_missing} or fewer missing PCIAT values: {rows_to_impute.sum()}")
    
    # Create copy of data for imputation
    pciat_data = train_df.loc[rows_to_impute, pciat_cols].copy()
    
    # Initialize KNN imputer
    imputer = Imputers(imputer_type=imputer_type, strategy=strategy, n_neighbors=n_neighbors, 
                       max_iter=max_iter, random_state=random_state)
    
    # Impute missing values only for rows with <=max_missing missing values
    imputed_data = imputer.fit_transform(pciat_data)
    
    # Create a copy of the input dataframe
    result_df = train_df.copy()
    
    # Update dataframe with imputed values
    result_df.loc[rows_to_impute, pciat_cols] = imputed_data

    # Get PCIAT columns excluding PCIAT-Season and PCIAT-PCIAT_Total
    pciat_cols = [col for col in result_df.columns if col.startswith('PCIAT') 
                and col != 'PCIAT-Season' 
                and col != 'PCIAT-PCIAT_Total']

    # Calculate sum of PCIAT columns, keeping NaN values as NaN
    result_df['PCIAT-recalc_total'] = result_df[pciat_cols].sum(axis=1, skipna=False)
    
    print(f"Successfully imputed PCIAT values for {rows_to_impute.sum()} rows")
    print(f"Number of rows with no missing PCIAT values after imputation: {result_df[pciat_cols].isnull().sum(axis=1).eq(0).sum()}")
    
    return result_df


def recalculate_sii_labels(train_df): # DO IT
    """
    Recalculate SII (Severity of Internet Addiction) labels based on PCIAT scores.
    
    Labels are assigned as follows:
    - 0: PCIAT total <= 30
    - 1: 31 <= PCIAT total <= 49
    - 2: 50 <= PCIAT total <= 79
    - 3: PCIAT total >= 80
    
    Takes into account maximum possible score when there are missing values.
    
    Args:
        train_df (pd.DataFrame): Input dataframe containing PCIAT columns
        
    Returns:
        pd.DataFrame: DataFrame with recalculated SII labels
    """
    # Create a copy of input dataframe
    result_df = train_df.copy()
    
    # Get PCIAT columns excluding special columns
    pciat_cols = [f'PCIAT-PCIAT_{i+1:02d}' for i in range(20)]
    
    # Calculate new SII labels
    def calculate_sii(row):
        if pd.isna(row['PCIAT-recalc_total']):
            return np.nan
            
        # Calculate maximum possible score considering missing values
        max_possible = row['PCIAT-recalc_total'] + row[pciat_cols].isna().sum() * 5
        
        # Determine SII label based on total score and maximum possible score
        if row['PCIAT-recalc_total'] <= 30 and max_possible <= 30:
            return 0
        elif 31 <= row['PCIAT-recalc_total'] <= 49 and max_possible <= 49:
            return 1
        elif 50 <= row['PCIAT-recalc_total'] <= 79 and max_possible <= 79:
            return 2
        elif row['PCIAT-recalc_total'] >= 80 and max_possible >= 80:
            return 3
        return np.nan
    
    # Apply calculation to each row
    result_df['recalc_sii'] = result_df.apply(calculate_sii, axis=1)
    
    # Drop old 'sii' column if it exists
    if 'sii' in result_df.columns:
        result_df = result_df.drop('sii', axis=1)
    
    # Remove rows with NaN recalc_sii values
    result_df = result_df.dropna(subset=['recalc_sii'])
    
    return result_df


def drop_season_columns(df): # ON - OFF
    # Drop all columns that contain 'Season' in their name
    return df.drop(columns=[col for col in df.columns if 'Season' in col])


def impute_physical_measurements(train_df, wh_imputer_type='knn', wc_imputer_type='linear_regression',
                               n_neighbors=5, random_state=42, strategy='mean', max_iter=10): # TRY DIFFERENT IMPUTERS
    """
    Impute missing physical measurement values (Weight, Height, and Waist Circumference).
    
    Args:
        train_df (pd.DataFrame): Input dataframe containing physical measurements
        wh_imputer_type (str): Type of imputer for Weight/Height ('knn', 'linear_regression' or others from Imputers class)
        wc_imputer_type (str): Type of imputer for Waist Circumference ('linear_regression' or others)
        n_neighbors (int): Number of neighbors for KNN imputation
        random_state (int): Random state for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with imputed physical measurements
    """
    # Create copy of input dataframe
    result_df = train_df.copy()
    
    # First impute Weight and Height
    features_for_wh = ['Basic_Demos-Age', 'Basic_Demos-Sex']
    target_wh = ['Physical-Weight', 'Physical-Height']
    
    # Initialize and fit imputer for Weight and Height
    wh_imputer = Imputers(imputer_type=wh_imputer_type, n_neighbors=n_neighbors, 
                         random_state=random_state, strategy=strategy, max_iter=max_iter)
    
    # Impute Weight and Height
    wh_imputed = wh_imputer.fit_transform(result_df[features_for_wh + target_wh])
    wh_imputed_df = pd.DataFrame(wh_imputed[:, -2:], columns=target_wh)
    
    # Update Weight and Height with imputed values
    for col in target_wh:
        result_df[col] = np.where(result_df[col].isna(),
                                 wh_imputed_df[col],
                                 result_df[col])
    
    # Recalculate BMI with new Weight and Height values
    result_df['Physical-BMI'] = result_df['Physical-Weight'] / ((result_df['Physical-Height'] / 100) ** 2)
    
    # Impute Waist Circumference
    waist_features = ['Physical-BMI', 'Physical-Weight']
    
    # Get rows with non-null values for training
    complete_rows = result_df[waist_features + ['Physical-Waist_Circumference']].dropna()
    
    if len(complete_rows) > 0:
        # Initialize imputer for Waist Circumference
        wc_imputer = Imputers(imputer_type=wc_imputer_type, random_state=random_state, 
                              strategy=strategy, max_iter=max_iter)
        
        # Fit and transform
        X_waist = complete_rows[waist_features]
        y_waist = complete_rows['Physical-Waist_Circumference']
        
        wc_imputer.fit(X_waist, y_waist)
        
        # Predict missing waist values
        missing_waist = result_df['Physical-Waist_Circumference'].isna()
        if missing_waist.any():
            X_predict = result_df.loc[missing_waist, waist_features]
            result_df.loc[missing_waist, 'Physical-Waist_Circumference'] = wc_imputer.predict(X_predict)
    
    print(f"Successfully imputed physical measurements")
    return result_df


def convert_to_metric_units(df):
    """
    Convert physical measurements from imperial to metric units.
    
    Conversions:
    - Weight: pounds to kilograms (x 0.453592)
    - Height: inches to centimeters (x 2.54)
    - Waist Circumference: inches to centimeters (x 2.54)
    - BMI: recalculated using metric units
    
    Args:
        df (pd.DataFrame): Input dataframe containing physical measurements
        
    Returns:
        pd.DataFrame: DataFrame with measurements in metric units
    """
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Conversion factors
    LBS_TO_KG = 0.453592
    INCHES_TO_CM = 2.54
    
    # Convert weight from pounds to kilograms
    if 'Physical-Weight' in result_df.columns:
        result_df['Physical-Weight'] = result_df['Physical-Weight'] * LBS_TO_KG
    
    # Convert height from inches to centimeters
    if 'Physical-Height' in result_df.columns:
        result_df['Physical-Height'] = result_df['Physical-Height'] * INCHES_TO_CM
    
    # Convert waist circumference from inches to centimeters
    if 'Physical-Waist_Circumference' in result_df.columns:
        result_df['Physical-Waist_Circumference'] = result_df['Physical-Waist_Circumference'] * INCHES_TO_CM
    
    # Recalculate BMI using metric units
    if all(col in result_df.columns for col in ['Physical-Weight', 'Physical-Height']):
        result_df['Physical-BMI'] = result_df.apply(
            lambda row: row['Physical-Weight'] / ((row['Physical-Height'] / 100) ** 2)
            if pd.notna(row['Physical-Weight']) and pd.notna(row['Physical-Height'])
            else np.nan,
            axis=1
        )
    
    return result_df


def check_bp_hr(df, check_pulse_pressure=None):
    """
    Validate and clean blood pressure and heart rate measurements.
    
    Applies the following checks:
    1. Replace 0 values with NaN
    2. Ensure systolic > diastolic
    3. Check for physiologically valid ranges:
        - Systolic: 70-250 mmHg
        - Diastolic: 40-130 mmHg
        - Heart Rate: 40-200 bpm
    4. Check for common BP ratios
    5. Check for digit preference patterns
    
    Args:
        df (pd.DataFrame): Input dataframe containing BP and HR measurements
        
    Returns:
        pd.DataFrame: DataFrame with cleaned BP and HR values
    """
    result_df = df.copy()
    
    bp_hr_cols = [
        'Physical-Diastolic_BP', 'Physical-Systolic_BP',
        'Physical-HeartRate'
    ]
    
    # Replace 0 values with NaN
    result_df[bp_hr_cols] = result_df[bp_hr_cols].replace(0, np.nan)
    
    # Basic relationship check: Systolic should be greater than Diastolic
    invalid_bp = result_df['Physical-Systolic_BP'] <= result_df['Physical-Diastolic_BP']
    result_df.loc[invalid_bp, bp_hr_cols] = np.nan
    
    # Physiologically valid ranges
    result_df.loc[result_df['Physical-Systolic_BP'] > 250, 'Physical-Systolic_BP'] = np.nan
    result_df.loc[result_df['Physical-Systolic_BP'] < 70, 'Physical-Systolic_BP'] = np.nan
    result_df.loc[result_df['Physical-Diastolic_BP'] > 130, 'Physical-Diastolic_BP'] = np.nan
    result_df.loc[result_df['Physical-Diastolic_BP'] < 40, 'Physical-Diastolic_BP'] = np.nan
    result_df.loc[result_df['Physical-HeartRate'] > 200, 'Physical-HeartRate'] = np.nan
    result_df.loc[result_df['Physical-HeartRate'] < 40, 'Physical-HeartRate'] = np.nan
    if check_pulse_pressure:    
        # Check for valid pulse pressure (systolic - diastolic difference)
        pulse_pressure = result_df['Physical-Systolic_BP'] - result_df['Physical-Diastolic_BP']
        invalid_pp = (pulse_pressure < 20) | (pulse_pressure > 100)
        result_df.loc[invalid_pp, ['Physical-Systolic_BP', 'Physical-Diastolic_BP']] = np.nan
    
    return result_df


def combine_paq_scores(df):
    """
    Combine PAQ-A and PAQ-C total scores into a single column.
    If both scores exist, takes their average. Otherwise, uses whichever score is available.
    
    Args:
        df (pd.DataFrame): Input dataframe containing PAQ-A and PAQ-C scores
        
    Returns:
        pd.DataFrame: DataFrame with combined PAQ scores and original columns dropped
    """
    # Create a copy of the input dataframe
    result_df = df.copy()
    
    # Define PAQ columns
    paq_cols = [
        'PAQ_A-PAQ_A_Total',
        'PAQ_C-PAQ_C_Total'
    ]
    
    # Combine scores
    result_df['PAQ_Total_Combined'] = np.where(
        result_df['PAQ_A-PAQ_A_Total'].notna() & result_df['PAQ_C-PAQ_C_Total'].notna(),  
        (result_df['PAQ_A-PAQ_A_Total'] + result_df['PAQ_C-PAQ_C_Total']) / 2,
        result_df['PAQ_A-PAQ_A_Total'].combine_first(result_df['PAQ_C-PAQ_C_Total'])
    )
    
    # Drop original PAQ columns
    result_df = result_df.drop(paq_cols, axis=1)
    
    return result_df


CONFIG = {
    'MIN_HOURS_PER_DAY': 4,
    'SAMPLES_PER_SECOND': 1/5,  # 5-second intervals
    'INACTIVITY_THRESHOLD': 0.02,
    'MIN_INACTIVITY_PERIOD': 720,
    'NUM_DAYS': 7,
    'BASE_FEATURES': ['x_std', 'y_std', 'z_std', 'anglez_std', 
                     'anglez_mean', 'light_mean', 'inactivity_duration']
}

def extract_daily_features(day_data, config=None):
    """Extract features from daily data using configurable parameters."""
    if config is None:
        config = CONFIG
        
    features = {}
    
    # Calculate all statistics at once
    numeric_cols = ['X', 'Y', 'Z', 'anglez', 'light']
    stats = day_data[numeric_cols].agg(['std', 'mean'])
    
    # Only include features specified in config
    for feature in config['BASE_FEATURES']:
        if feature == 'inactivity_duration':
            continue
        stat_type = feature.split('_')[-1]  # 'mean' or 'std'
        col_name = feature.replace(f'_{stat_type}', '')
        features[feature] = stats.loc[stat_type, col_name]
    
    return features

def calculate_inactivity_duration(day_data: pd.DataFrame, config=None) -> float:
    if config is None:
        config = CONFIG
        
    try:
        if day_data.empty:
            return 0.0
            
        if 'enmo' not in day_data.columns:
            print(f"Missing required columns. Available columns: {day_data.columns.tolist()}")
            return 0.0

        is_inactive = day_data['enmo'] < config['INACTIVITY_THRESHOLD']
        
        inactivity_changes = np.diff(np.concatenate(([0], is_inactive.values, [0])))
        inactivity_starts = np.where(inactivity_changes == 1)[0]
        inactivity_ends = np.where(inactivity_changes == -1)[0]
        
        if len(inactivity_starts) == 0 or len(inactivity_ends) == 0:
            return 0.0
        
        total_inactivity_time = 0
        
        for start, end in zip(inactivity_starts, inactivity_ends):
            if start >= len(day_data) or end >= len(day_data):
                continue
                
            period_length = end - start
            if period_length >= config['MIN_INACTIVITY_PERIOD']:
                total_inactivity_time += period_length
        
        # Convert to hours using sampling rate
        day_inactivity_time = (total_inactivity_time * (1/config['SAMPLES_PER_SECOND'])) / 3600
        
        return day_inactivity_time
        
    except Exception as e:
        print(f"Error processing day: {str(e)}")
        return 0.0

def process_single_day(data, day_start_idx, samples_per_day, config=None):
    if config is None:
        config = CONFIG
        
    day_end_idx = min(day_start_idx + samples_per_day, len(data))
    day_data = data.iloc[day_start_idx:day_end_idx]
    
    min_samples = (config['MIN_HOURS_PER_DAY'] * 60 * 60) * config['SAMPLES_PER_SECOND']
    if len(day_data) < min_samples:
        return None
        
    try:
        features = extract_daily_features(day_data, config)
        inactivity_duration = calculate_inactivity_duration(day_data, config)
        features['inactivity_duration'] = inactivity_duration
        return features
    except Exception as e:
        print(f"Error processing day: {str(e)}")
        return None

def process_file(file_id: str, dirname: str) -> Tuple[Optional[List], Optional[List], str]:
    try:
        file_path = os.path.join(dirname, file_id)
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            return None, None, file_id.replace('id=', '')
            
        data = pd.read_parquet(file_path)
        
        # Add validation check here
        if not validate_data(data):
            logging.warning(f"Invalid data in file: {file_path}")
            return None, None, file_id.replace('id=', '')
        
        if data.empty:
            logging.warning(f"Empty file: {file_path}")
            return None, None, file_id.replace('id=', '')
        
        samples_per_day = 24 * 60 * 60 // 5
        total_days = len(data) // samples_per_day + (1 if len(data) % samples_per_day > 0 else 0)
        
        daily_features = []
        daily_stats = []
        
        for day in range(total_days):
            day_features = process_single_day(data, day * samples_per_day, samples_per_day)
            if day_features is None:
                continue
            daily_features.append(day_features)
            daily_stats.append(day_features.pop('stats'))
        
        return daily_stats, daily_features, file_id.replace('id=', '')

    except Exception as e:
        print(f"Error processing file {file_id}: {str(e)}")
        return None, None, file_id.replace('id=', '')

def load_time_series(dirname: str, config: dict = None, chunk_size: int = 100) -> pd.DataFrame:
    if config is None:
        config = CONFIG
        
    temp_df = pd.read_parquet(dirname, columns=['id'])
    ids = temp_df['id'].unique()
    
    # Process in chunks to manage memory
    chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    all_features = []
    
    for chunk in chunks:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
            futures = [executor.submit(process_file, f"id={id_}", dirname) 
                      for id_ in chunk]
            
            for future in tqdm(futures, desc=f"Processing chunk"):
                result = future.result()
                if result[1] is not None:
                    all_features.append((result[1], result[2]))
    
    if not all_features:
        raise ValueError("No valid features extracted")
    
    # Calculate number of days from first record
    num_days = config['NUM_DAYS']  # We want 7 days of features
    
    # Create feature names
    base_features = config["BASE_FEATURES"]
    feature_names = []
    for day in range(num_days):
        for feat in base_features:
            feature_names.append(f"{feat}_day{day+1}")
    
    # Flatten and prepare results
    results = []
    for features, id_str in all_features:
        flat_features = []
        for day_features in features[:num_days]:  # Ensure consistent length
            for feat in base_features:
                flat_features.append(day_features.get(feat, 0.0))
        results.append((flat_features, id_str))
    
    # Create DataFrame
    features, indexes = zip(*results)
    df = pd.DataFrame(features, columns=feature_names)
    df['id'] = indexes
    
    return df

def validate_data(data: pd.DataFrame) -> bool:
    """Validate input data meets requirements."""
    required_columns = {'X', 'Y', 'Z', 'anglez', 'light', 'enmo'}
    
    if not all(col in data.columns for col in required_columns):
        missing = required_columns - set(data.columns)
        logging.error(f"Missing required columns: {missing}")
        return False
        
    if data.empty:
        logging.error("Empty dataset provided")
        return False
        
    return True


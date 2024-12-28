from preprocessing import *
from trainer import TrainML

def load_tabular_data(path):
    return pd.read_csv(path)   

def prepare_data(tabular_data=None, actiography_data=None, use_both_data=None, random_state=42, drop_season=True, convert_to_metric=True):
    if tabular_data:
        tabular_data = load_tabular_data(tabular_data)
    if actiography_data:
        actiography_data = load_time_series(actiography_data)


    tabular_data = impute_pciat_values(tabular_data, imputer_type='knn', max_missing=5, n_neighbors=5)

    tabular_data = recalculate_sii_labels(tabular_data)

    if drop_season:
        tabular_data = drop_season_columns(tabular_data)

    tabular_data = impute_physical_measurements(tabular_data, wh_imputer_type='knn', wc_imputer_type='linear_regression')

    if convert_to_metric:
        tabular_data = convert_to_metric_units(tabular_data)

    tabular_data = check_bp_hr(tabular_data, check_pulse_pressure=None)

    tabular_data = combine_paq_scores(tabular_data)

    if use_both_data:
        tabular_data = tabular_data.merge(actiography_data, on='id', how='left')

    return tabular_data

def train_model(tabular_data, random_state=42, selected_models=None):

    target_columns = [col for col in tabular_data.columns if 'PCIAT' in col or 'sii' in col]

    trainer = TrainML(n_splits=5, random_state=random_state)

    target_cols = [item for item in target_columns if item not in ['sii', 'PCIAT-Season']]
    main_target = 'recalc_sii'

    trainer.cross_validate(tabular_data, target_cols, main_target)
    trainer.plot_confusion_matrix()
    trainer.plot_overall_confusion_matrix()

    print("Fold Results:", trainer.fold_results)
    print("Best Fold Score:", trainer.best_fold_score)
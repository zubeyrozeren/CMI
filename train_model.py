from preprocessing import *
from trainer import TrainML



def prepare_data(tabular_data=None, actiography_data=None, use_both_data=None, random_state=42, drop_season=True, convert_to_metric=True):
    if tabular_data:
        print("Loading tabular data...")
        tabular_data = pd.read_csv(tabular_data)

    if actiography_data:
        print("Loading actiography data...")
        actiography_data = load_time_series(actiography_data)

    print("Imputing PCIAT values...")
    tabular_data = impute_pciat_values(tabular_data, imputer_type='knn', max_missing=5, n_neighbors=5)

    print("Recalculating SII labels...")
    tabular_data = recalculate_sii_labels(tabular_data)

    if drop_season:
        print("Dropping season columns...")
        tabular_data = drop_season_columns(tabular_data)

    print("Imputing physical measurements...")
    tabular_data = impute_physical_measurements(tabular_data, wh_imputer_type='knn', wc_imputer_type='linear_regression')

    if convert_to_metric:
        print("Converting to metric units...")
        tabular_data = convert_to_metric_units(tabular_data)

    print("Checking BP and HR...")
    tabular_data = check_bp_hr(tabular_data, check_pulse_pressure=None)

    print("Combining PAQ scores...")
    tabular_data = combine_paq_scores(tabular_data)

    if use_both_data:
        print("Merging actiography data...")
        tabular_data = tabular_data.merge(actiography_data, on='id', how='left')

    return tabular_data

def train_model(tabular_data, random_state=42, selected_models=None):
    print("Preparing data...")
    target_columns = [col for col in tabular_data.columns if 'PCIAT' in col or 'sii' in col]

    print("Training model...")
    trainer = TrainML(n_splits=5, random_state=random_state)

    target_cols = [item for item in target_columns if item not in ['sii', 'PCIAT-Season']]
    main_target = 'recalc_sii'

    print("Cross-validating model...")
    trainer.cross_validate(tabular_data, target_cols, main_target)
    trainer.plot_confusion_matrix()
    trainer.plot_overall_confusion_matrix()

    print("Fold Results:", trainer.fold_results)
    print("Best Fold Score:", trainer.best_fold_score)



if __name__ == "__main__":
    data = prepare_data("data/train.csv")
    train_model(data)
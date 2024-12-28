from imputers import Imputers
from scalers import Scalers
from models import Models
from sklearn.ensemble import VotingRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np

class TrainML:
    DEFAULT_CONFIGS = {
        'lgbm': {
            'objective': 'poisson',
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.6,
            'colsample_bytree': 0.5,
            'min_data_in_leaf': 100,
            'random_state': 42,
            'verbosity': -1
        },
        'catboost': {
            'objective': 'RMSE',
            'iterations': 250,
            'depth': 4,
            'learning_rate': 0.05,
            'l2_leaf_reg': 0.09,
            'bagging_temperature': 0.3,
            'random_strength': 3.5,
            'min_data_in_leaf': 60,
            'random_state': 42,
            'verbose': 0
        },
        'gaussian_nb': {
            'var_smoothing': 1e-9,
            'priors': np.array([0.58, 0.26, 0.13, 0.03])
        },
        'logistic_regression': {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'multi_class': 'multinomial',
            'class_weight': 'balanced'
        }
    }

    def __init__(self, n_splits=5, random_state=42, thresholds=None):
        self.n_splits = n_splits
        self.random_state = random_state

        # Use Scalers class
        self.scaler = Scalers("minmax")
        
        # Use Imputers class
        self.imputer = Imputers(
            imputer_type='iterative',
            max_iter=5,
            random_state=random_state
        )

        self.fold_results = []
        self.thresholds = thresholds if thresholds is not None else [0.5, 1.5, 2.5]

        self.best_fold_true = None
        self.best_fold_pred = None
        self.best_fold_score = 0
        self.all_fold_true = []
        self.all_fold_pred = []

        self.feature_importances = {}
        self.fold_feature_importances = []

        # Initialize models directly with their configurations
        self.lr_model = LogisticRegression(**self.DEFAULT_CONFIGS['logistic_regression'])
        self.nb_model = GaussianNB(**self.DEFAULT_CONFIGS['gaussian_nb'])
        
        # Initialize ensemble models
        LGBM_Model = LGBMRegressor(**self.DEFAULT_CONFIGS['lgbm'])
        XGB_Model = XGBRegressor(random_state=random_state)
        CatBoost_Model = CatBoostRegressor(**self.DEFAULT_CONFIGS['catboost'])
        
        # Create ensemble
        self.ensemble_model = VotingRegressor(
            estimators=[
                ('lightgbm', LGBM_Model),
                #('xgboost', XGB_Model),
                ('catboost', CatBoost_Model)
            ]
        )

    def drop_features(self, df):
        features_to_drop = [
            'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference', 
            'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',

            'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',

            'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND', 'FGC-FGC_GSND_Zone',
            'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU', 'FGC-FGC_PU_Zone',
            'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone',
            'FGC-FGC_TL', 'FGC-FGC_TL_Zone',
                
            'BIA-BIA_BMC', 'BIA-BIA_ECW', 'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_TBW'

            'BSA'
        ]
            
        existing_features = [col for col in features_to_drop if col in df.columns]
        df = df.drop(columns=existing_features)
            
        return df

    def feature_engineering(self, df):
        def safe_divide(a, b, fill_value=0):
            result = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))
            return np.clip(result, -1e15, 1e15)

        # Physical measurements ratios
        """df['Waist_to_Height'] = safe_divide(df['Physical-Waist_Circumference'], df['Physical-Height'])
        
        # Blood pressure and heart rate derivatives
        df['Pulse_Pressure'] = df['Physical-Systolic_BP'] - df['Physical-Diastolic_BP']
        df['Mean_Arterial_Pressure'] = (2 * df['Physical-Diastolic_BP'] + df['Physical-Systolic_BP']) / 3
        
        # Body surface area calculations
        df['BSA'] = np.sqrt((df['Physical-Height'] * df['Physical-Weight']) / 3600)
        df['HR_to_BSA'] = safe_divide(df['Physical-HeartRate'], df['BSA'])

        # Fitness zone aggregation
        fitness_zones = [
            'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone', 
            'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 
            'FGC-FGC_TL_Zone'
        ]
        df['Overall_Fitness_Score'] = df[fitness_zones].mean(axis=1)

        # Fitness and endurance metrics
        df['Max_Aerobic_Cap'] = safe_divide(
            df['Fitness_Endurance-Max_Stage'], 
            df['Fitness_Endurance-Time_Mins']
        )

        # Screen time and activity ratios
        df['Screen_vs_Activity'] = safe_divide(
            df['PreInt_EduHx-computerinternet_hoursday'], 
            df['PAQ_Total_Combined']
        )
        df['Age_Adjusted_Usage'] = safe_divide(
            df['PreInt_EduHx-computerinternet_hoursday'], 
            df['Basic_Demos-Age']
        )

        # Body composition ratios
        body_comp_features = {
            'Muscle_to_Fat': ('BIA-BIA_SMM', 'BIA-BIA_FMI'),
            'BMI_to_Age': ('Physical-BMI', 'Basic_Demos-Age'),
            'Hydration_Status': ('BIA-BIA_TBW', 'Physical-Weight'),
            'ICW_TBW': ('BIA-BIA_ICW', 'BIA-BIA_TBW'),
            'SMM_Height': ('BIA-BIA_SMM', 'Physical-Height'),
            'BFP_BMI': ('BIA-BIA_Fat', 'BIA-BIA_BMI'),
            'FFMI_BFP': ('BIA-BIA_FFMI', 'BIA-BIA_Fat'),
            'FMI_BFP': ('BIA-BIA_FMI', 'BIA-BIA_Fat'),
            'LST_TBW': ('BIA-BIA_LST', 'BIA-BIA_TBW')
        }

        for feature_name, (numerator, denominator) in body_comp_features.items():
            df[feature_name] = safe_divide(df[numerator], df[denominator])

        # Metabolic and energy expenditure features
        energy_features = {
            'BMR_Weight': ('BIA-BIA_BMR', 'Physical-Weight'),
            'DEE_Weight': ('BIA-BIA_DEE', 'Physical-Weight')
        }

        for feature_name, (numerator, denominator) in energy_features.items():
            df[feature_name] = safe_divide(df[numerator], df[denominator])

        # Interaction features
        df['BMI_Age'] = df['Physical-BMI'] * df['Basic_Demos-Age']
        df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
        df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
        df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
        df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']

        # Clinical assessment interactions
        df['CGAS_Internet_Impact'] = safe_divide(
            df['CGAS-CGAS_Score'], 
            df['PreInt_EduHx-computerinternet_hoursday']
        )"""

        # Clean up any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop unnecessary features
        df = self.drop_features(df)
        
        return df

    def preprocess(self, df, fit=True):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except ValueError:
                df[col] = pd.Categorical(df[col]).codes

        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
        
        if fit:
            df_scaled = self.scaler.fit_transform(df[numerical_features])
            df_imputed = self.imputer.fit_transform(df_scaled)
        else:
            df_scaled = self.scaler.transform(df[numerical_features])
            df_imputed = self.imputer.transform(df_scaled)

        df[numerical_features] = df_imputed
        return df
    
    def derive_sii(self, df, thresholds):
        result = np.digitize(df, bins=thresholds)
        return result

    def analyze_feature_importance(self, X, fold_idx):
        importance_dict = {}
        
        # Change self.model to self.ensemble_model
        for name, model in self.ensemble_model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = dict(zip(X.columns, model.feature_importances_))
        
        return importance_dict

    def cross_validate(self, df, target_cols, main_target):
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        X = df.drop(columns=target_cols + [main_target])
        y = df[main_target]
        fold_idx = 1
        
        all_features = set()
        
        for train_index, test_index in kf.split(X, y):
            print(f"\n=== Processing Fold {fold_idx}/{self.n_splits} ===")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train = self.feature_engineering(X_train)
            X_test = self.feature_engineering(X_test)

            all_features.update(X_train.columns)

            X_train = self.preprocess(X_train, fit=True)
            X_test = self.preprocess(X_test, fit=False)

            # Train all models
            self.ensemble_model.fit(X_train, y_train)
            y_train_class = self.derive_sii(y_train, self.thresholds)
            self.nb_model.fit(X_train, y_train_class)
            self.lr_model.fit(X_train, y_train_class)

            # Get predictions from all models
            y_pred_reg = self.ensemble_model.predict(X_test)
            y_pred_class_ensemble = self.derive_sii(y_pred_reg, self.thresholds)
            y_pred_class_nb = self.nb_model.predict(X_test)
            y_pred_class_lr = self.lr_model.predict(X_test)
            
            # Get probabilities
            nb_probabilities = self.nb_model.predict_proba(X_test)
            lr_probabilities = self.lr_model.predict_proba(X_test)

            # Start with ensemble predictions
            y_pred_class = np.round((y_pred_class_ensemble + y_pred_class_lr + y_pred_class_nb) / 3)

            y_test_class = self.derive_sii(y_test, self.thresholds)

            # Calculate and store metrics
            fold_importance = self.analyze_feature_importance(X_train, fold_idx)
            self.fold_feature_importances.append(fold_importance)

            fold_f1 = f1_score(y_test_class, y_pred_class, average='weighted')
            fold_mcc = matthews_corrcoef(y_test_class, y_pred_class)  # Add this line
            self.fold_results.append({'f1': fold_f1, 'mcc': fold_mcc})  # Modified to store both metrics

            self.all_fold_true.extend(y_test_class)
            self.all_fold_pred.extend(y_pred_class)

            if fold_f1 > self.best_fold_score:
                self.best_fold_score = fold_f1
                self.best_fold_true = y_test_class
                self.best_fold_pred = y_pred_class

            # Print individual model performances
            print(f"\nFold {fold_idx} Performance:")
            print(f"Combined F1-Score: {fold_f1:.4f}")
            print(f"Combined MCC-Score: {fold_mcc:.4f}")  # Add this line
            print(f"Ensemble F1-Score: {f1_score(y_test_class, y_pred_class_ensemble, average='weighted'):.4f}")
            print(f"Ensemble MCC-Score: {matthews_corrcoef(y_test_class, y_pred_class_ensemble):.4f}")  # Add this line
            print(f"NaiveBayes F1-Score: {f1_score(y_test_class, y_pred_class_nb, average='weighted'):.4f}")
            print(f"NaiveBayes MCC-Score: {matthews_corrcoef(y_test_class, y_pred_class_nb):.4f}")  # Add this line
            print(f"LogisticRegression F1-Score: {f1_score(y_test_class, y_pred_class_lr, average='weighted'):.4f}")
            print(f"LogisticRegression MCC-Score: {matthews_corrcoef(y_test_class, y_pred_class_lr):.4f}")  # Add this line

            fold_idx += 1

            self.aggregate_feature_importance()
            

            print(f"\n=== Cross-Validation Summary ===")
            print(f"Mean F1-Score: {np.mean([r['f1'] for r in self.fold_results]):.4f}")
            print(f"F1-Score Std Dev: {np.std([r['f1'] for r in self.fold_results]):.4f}")
            print(f"Mean MCC-Score: {np.mean([r['mcc'] for r in self.fold_results]):.4f}")
            print(f"MCC-Score Std Dev: {np.std([r['mcc'] for r in self.fold_results]):.4f}")
            
            self.print_feature_importance_summary()

    def aggregate_feature_importance(self):
        aggregated = {}
        
        for fold_importance in self.fold_feature_importances:
            for model_name, importances in fold_importance.items():
                if model_name not in aggregated:
                    aggregated[model_name] = {}
                
                for feature, importance in importances.items():
                    if feature not in aggregated[model_name]:
                        aggregated[model_name][feature] = []
                    aggregated[model_name][feature].append(importance)
        
        self.feature_importances = {
            model_name: {
                feature: {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                for feature, values in feature_dict.items()
            }
            for model_name, feature_dict in aggregated.items()
        }

    def print_feature_importance_summary(self):
        print("\n=== Feature Importance Analysis ===")
        
        for model_name, features in self.feature_importances.items():
            print(f"\nModel: {model_name}")
            print("-" * 60)
            print(f"{'Feature':<30} {'Mean Importance':>15} {'Std Dev':>15}")
            print("-" * 60)
            
            sorted_features = sorted(
                features.items(),
                key=lambda x: x[1]['mean'],
                reverse=True
            )
            
            for feature, stats in sorted_features[:20]:
                print(f"{feature[:30]:<30} {stats['mean']:>15.4f} {stats['std']:>15.4f}")
    
    def get_top_features(self, n_features=20):
        top_features = {}
        
        for model_name, features in self.feature_importances.items():
            sorted_features = sorted(
                features.items(),
                key=lambda x: x[1]['mean'],
                reverse=True
            )
            top_features[model_name] = {
                'features': [f[0] for f in sorted_features[:n_features]],
                'importance': [f[1]['mean'] for f in sorted_features[:n_features]]
            }
            
        return top_features

    def plot_confusion_matrix(self):
            if self.best_fold_true is None or self.best_fold_pred is None:
                print("No fold data available. Run cross_validate first.")
                return

            unique_labels = np.unique(np.concatenate([self.best_fold_true, self.best_fold_pred]))
            cm = confusion_matrix(self.best_fold_true, self.best_fold_pred, labels=unique_labels)

            plt.figure(figsize=(10,7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=unique_labels)
            disp.plot(cmap='viridis')
            plt.title(f'Karışıklık Matrisi (En İyi Fold, F1-Skoru: {self.best_fold_score:.4f})')
            plt.xlabel('Tahmin')
            plt.ylabel('Gerçek')
            plt.tight_layout()

            plt.savefig('best_fold_confusion.png', dpi=600)  # High resolution and transparent background
            #plt.show()
            

    def plot_overall_confusion_matrix(self):
        if not self.all_fold_true or not self.all_fold_pred:
            print("No data available. Run cross_validate first.")
            return

        unique_labels = np.unique(np.concatenate([self.all_fold_true, self.all_fold_pred]))
        cm = confusion_matrix(self.all_fold_true, self.all_fold_pred, labels=unique_labels)

        plt.figure(figsize=(10, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap='viridis')
        plt.title('Overall Confusion Matrix (All Folds)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        plt.savefig('confusion_matrix.png', dpi=600)  # High resolution and transparent background
        #plt.show()

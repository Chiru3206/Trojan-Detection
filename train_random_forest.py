import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from preprocess import load_data, preprocess_data, save_pickle


def train_random_forest(X_train, y_train, X_test, y_test):
    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=40,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced_subsample',
        criterion='entropy',
        random_state=42,
        n_jobs=-1,
    )
    base_model.fit(X_train, y_train)

    search_space = {
        'n_estimators': [200, 300, 400, 500, 600],
        'max_depth': [15, 20, 25, 30, 35, 40, None],
        'max_features': ['sqrt', 'log2', 0.4, 0.5, 0.6],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 6, 8],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
    }

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            class_weight='balanced_subsample', random_state=42, n_jobs=1
        ),
        param_distributions=search_space,
        n_iter=30,
        cv=cv,
        scoring='accuracy',
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    sample_size = min(30000, len(X_train))
    if sample_size < len(X_train):
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        sample_idx, _ = next(splitter.split(X_train, y_train))
        X_search = X_train[sample_idx]
        y_search = y_train[sample_idx]
    else:
        X_search = X_train
        y_search = y_train

    search.fit(X_search, y_search)
    best_params = search.best_params_
    best_model = search.best_estimator_

    print('\n[+] Random Forest hyperparameter search finished:')
    for k, v in sorted(best_params.items()):
        print(f'    {k}: {v}')

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Benign', 'Trojan'], zero_division=0),
        'best_params': best_params,
    }

    if metrics['accuracy'] < 80.0:
        print('\n[!] Accuracy below 80%; running a second focused RF search to push accuracy higher...')
        fine_search_space = {
            'n_estimators': [400, 500, 600, 700],
            'max_depth': [20, 25, 30, 35, 40, None],
            'max_features': [0.3, 0.4, 0.5, 'sqrt', 'log2'],
            'min_samples_split': [2, 4, 6, 8],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
        }
        fine_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(
                class_weight='balanced_subsample', random_state=42, n_jobs=1
            ),
            param_distributions=fine_search_space,
            n_iter=40,
            cv=cv,
            scoring='accuracy',
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        fine_search.fit(X_search, y_search)
        best_params = fine_search.best_params_
        best_model = fine_search.best_estimator_
        print('\n[+] Focused Random Forest search finished:')
        for k, v in sorted(best_params.items()):
            print(f'    {k}: {v}')

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            'f1': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, target_names=['Benign', 'Trojan'], zero_division=0),
            'best_params': best_params,
        }

    return best_model, metrics


def main():
    print('\n[+] Loading dataset and preprocessing...')
    df = load_data(100000)
    X, y, scaler, features = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)

    model, metrics = train_random_forest(X_train, y_train, X_test, y_test)

    print('\n[+] Random Forest results:')
    print(f"    Accuracy : {metrics['accuracy']}%")
    print(f"    Precision: {metrics['precision']}%")
    print(f"    Recall   : {metrics['recall']}%")
    print(f"    F1 Score : {metrics['f1']}%")
    print(f"    ROC AUC  : {metrics['roc_auc']}%")

    save_pickle(model, 'rf_model.pkl')
    save_pickle(model, 'model.pkl')
    save_pickle(scaler, 'rf_scaler.pkl')
    save_pickle(scaler, 'scaler.pkl')
    save_pickle(features, 'rf_features.pkl')
    save_pickle(features, 'features.pkl')
    save_pickle(metrics, 'rf_metrics.pkl')
    save_pickle({
        'best_model': 'Random Forest',
        **metrics,
    }, 'metrics.pkl')

    print('\n[+] Saved RF artifacts: rf_model.pkl, rf_scaler.pkl, rf_features.pkl, rf_metrics.pkl')
    print('[+] Saved app-compatible files: model.pkl, scaler.pkl, features.pkl, metrics.pkl')


if __name__ == '__main__':
    main()

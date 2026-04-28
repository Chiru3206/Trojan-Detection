import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from preprocess import load_data, preprocess_data, save_pickle


def main():
    print('\n[+] Loading dataset and preprocessing...')
    df = load_data(100000)
    X, y, scaler, features = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)

    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Benign', 'Trojan'], zero_division=0),
    }

    save_pickle(model, 'knn_model.pkl')
    save_pickle(scaler, 'knn_scaler.pkl')
    save_pickle(features, 'knn_features.pkl')
    save_pickle(metrics, 'knn_metrics.pkl')

    print('\n[+] KNN results:')
    print(f"    Accuracy : {metrics['accuracy']}%")
    print(f"    Precision: {metrics['precision']}%")
    print(f"    Recall   : {metrics['recall']}%")
    print(f"    F1 Score : {metrics['f1']}%")
    print(f"    ROC AUC  : {metrics['roc_auc']}%")
    print('\n[+] Saved KNN artifacts: knn_model.pkl, knn_scaler.pkl, knn_features.pkl, knn_metrics.pkl')


if __name__ == '__main__':
    main()

import sys

from train_random_forest import main as train_random_forest
from train_knn import main as train_knn
from train_cnn import main as train_cnn


def usage():
    print('Usage: python train_models.py [rf|knn|cnn|all]')
    print('\nOptions:')
    print('  rf   - Train only the Random Forest model')
    print('  knn  - Train only the KNN model')
    print('  cnn  - Train only the CNN model')
    print('  all  - Train RF, KNN, and CNN sequentially')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == 'rf':
        train_random_forest()
    elif cmd == 'knn':
        train_knn()
    elif cmd == 'cnn':
        train_cnn()
    elif cmd == 'all':
        train_random_forest()
        train_knn()
        train_cnn()
    else:
        usage()
        sys.exit(1)

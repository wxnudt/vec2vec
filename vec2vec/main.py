
import argparse
import logging
import os
import sys
import datetime
from sklearn.preprocessing import scale as scale_fun
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets as ds
import vec2vec.matrix2vec as vm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.append(rootPath)

def parse_args():
    '''
    Parses the matrix2vec arguments.
    (matrix, dimensions, num_walks=10, walk_length=20, window_size=10, topk=10, p=1, q=1, num_iter=20)
    '''
    parser = argparse.ArgumentParser(description="Run matrix2vec.")

    parser.add_argument('--input', nargs='?', default='./data/train.bow',
                        help='Input matrix file path')


    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')

    # parser.add_argument('--num_walks', type=int, default=10,
    #                   help='Number of walks per source. Default is 10.')

    parser.add_argument('--walk_length', type=int, default=20,
                        help='Length of walk per source. Default is 20.')

    parser.add_argument('--window_size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--topk', default=10, type=int,
                        help='Number of topk. Default is 10.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyper-parameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyper-parameter. Default is 1.')

    parser.add_argument('--num_iter', type=int, default=20,
                        help='Number of iterations. Default is 20.')

    return parser.parse_args()


def main(args):
    """
    Reduce the dimension of the matrix
    """

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    x_train2, y_train2 = ds.load_svmlight_file(args.input)
    x_train = x_train2.toarray()
    y_train = y_train2

    x_train = x_train[0:2000, :]
    y_train = y_train[0:2000]

    matrix = x_train
    dimensions = args.dimensions
    # num_walks = args.num_walks
    walk_length = args.walk_length
    window_size = args.window_size
    topk = args.topk
    p = args.p
    q = args.q
    num_iter = args.num_iter

    for num_walks in range(5, 55, 5):
        print("************* The number of num_walks is : " + str(num_walks) + " *******************")
        start = datetime.datetime.now()

        X_transformed = vm.matrix2vec(matrix, dimensions, num_walks, walk_length, window_size, topk, p, q,
                                              num_iter)

        end = datetime.datetime.now()

        # scale
        X_transformed = scale_fun(X_transformed)

        print('Model Matrix2vec Finished in ' + str(end - start) + " s.")

        # Using KNN classifier to test the result with cross_validation
        x_tr, x_te, y_tr, y_te = train_test_split(X_transformed, y_train, test_size=0.25)
        knn = KNeighborsClassifier()
        param = {"n_neighbors": [1, 3, 5, 7, 11]}  # 构造一些参数的值进行搜索 (字典类型，可以有多个参数)
        gc = GridSearchCV(knn, param_grid=param, cv=4)
        gc.fit(X_transformed, y_train)
        knn = gc.best_estimator_
        scores = cross_val_score(knn, X_transformed, y_train, cv=4)
        print("交叉验证Accuracy： ", scores)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


if __name__ == "__main__":
    args = parse_args()
    main(args)


def vec2vec(input="./data/train.bow", dimensions=64, walk_length=20, window_size=10, topk=10, p=1, q=1, num_iter=20):
    """
    Reduce the dimension of the matrix
    """

    x_train2, y_train2 = ds.load_svmlight_file(input)
    x_train = x_train2.toarray()
    y_train = y_train2

    x_train = x_train[0:2000, :]
    y_train = y_train[0:2000]

    matrix = x_train

    for num_walks in range(5, 55, 5):
        print("************* The number of num_walks is : " + str(num_walks) + " *******************")
        start = datetime.datetime.now()

        X_transformed = vm.matrix2vec(matrix, dimensions, num_walks, walk_length, window_size, topk, p, q,
                                              num_iter)

        end = datetime.datetime.now()

        # scale
        X_transformed = scale_fun(X_transformed)

        print('Model Matrix2vec Finished in ' + str(end - start) + " s.")

        # Using KNN classifier to test the result with cross_validation
        x_tr, x_te, y_tr, y_te = train_test_split(X_transformed, y_train, test_size=0.25)
        knn = KNeighborsClassifier()
        param = {"n_neighbors": [1, 3, 5, 7, 11]}  # 构造一些参数的值进行搜索 (字典类型，可以有多个参数)
        gc = GridSearchCV(knn, param_grid=param, cv=4)
        gc.fit(X_transformed, y_train)
        knn = gc.best_estimator_
        scores = cross_val_score(knn, X_transformed, y_train, cv=4)
        print("交叉验证Accuracy： ", scores)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

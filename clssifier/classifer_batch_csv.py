from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC

import scipy.io as sio
import numpy as np
import pandas as pd
import os
import time
import pickle
import warnings
import csv


# warnings.filterwarnings('ignore')

# import classifier_main_batch


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


# KNN Classifier
def knn_classifier(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(X_train, y_train):
    from sklearn.linear_model import LogisticRegressionCV
    model = LogisticRegressionCV(penalty='l2')
    model.fit(X_train, y_train)
    return model


# Random Forest Classifier
def random_forest_classifier(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=68, random_state=2)
    model.fit(X_train, y_train)
    return model


# Decision Tree Classifier
def decision_tree_classifier(X_train, y_train):
    from sklearn import tree
    model = tree.DecisionTreeClassifier(random_state=2)
    model.fit(X_train, y_train)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200, random_state=2)
    model.fit(X_train, y_train)
    return model


# SVM Classifier
def svm_classifier(X_train, y_train):
    model = SVC(kernel='rbf', probability=True, random_state=2)
    model.fit(X_train, y_train)
    return model


def classifier_1(X_train, X_test, y_train, y_test, save_dir, name):
    model_save_file = None
    model_save = {}
    # test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'SVM']
    num = ['1', '2', '3', '4', '5']
    classifiers = {
        # 'NB': naive_bayes_classifier,
        'KNN': knn_classifier,
        'LR': logistic_regression_classifier,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier,
        'SVM': svm_classifier,
        # 'GBDT': gradient_boosting_classifier
    }

    print('reading training and testing data...')
    # X_train, y_train, X_test, y_test = read_data(data_file)

    is_binary_class = (len(np.unique(y_train)) == 2)

    for classifier, i in zip(test_classifiers, num):
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](X_train, y_train)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(X_test)

        probility = model.predict_proba(X_test)
        dic = {name + '_%s_score' % classifier: probility}
        sio.savemat(save_dir + name + '_' + classifier + '_score' + '.mat', dic)
        if model_save_file != None:
            model_save[classifier] = model
        accuracy = metrics.accuracy_score(y_test, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))
        accuracy_dict[classifier] = 100 * accuracy

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))


def classifier_NB(X_train, X_test, y_train, y_test, save_dir, name):
    # global accuracy_dict
    print("******************* NB ********************")
    is_binary_class = (len(np.unique(y_train)) == 2)

    print('******************* %s ********************' % naive_bayes_classifier)
    start_time = time.time()
    model = naive_bayes_classifier(X_train, y_train)
    print('training took %fs!' % (time.time() - start_time))
    predict = model.predict(X_test)
    probility = model.predict_proba(X_test)
    dic = {name + '_%s_score' % 'NB': probility}
    sio.savemat(save_dir + name + '_NB_score' + '.mat', dic)
    accuracy = metrics.accuracy_score(y_test, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))
    accuracy_dict['NB'] = 100 * accuracy


def classifer_adaboost(X_train, X_test, y_train, y_test, save_dir, name):
    # global accuracy_dict
    print("******************* adaboost ********************")
    start_time = time.time()
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=5),
                               algorithm="SAMME",
                               n_estimators=500, learning_rate=0.7, random_state=2)
    model.fit(X_train, y_train)

    predict = model.predict(X_test)  # predict is holped to be equal to y_test
    accuracy = metrics.accuracy_score(y_test, predict)
    cost_time = time.time() - start_time
    print('accuracy: %.2f%%' % (100 * accuracy))
    print("cost time:", cost_time, "(s)......")
    probility = model.predict_proba(X_test)
    accuracy_dict['adaboost'] = 100 * accuracy
    dic = {name + '_adaboost_score': probility}
    sio.savemat(save_dir + name + '_adaboost_score' + '.mat', dic)


def classifier_GBDT(X_train, X_test, y_train, y_test, save_dir, name):
    # global accuracy_dict
    true_dict = {}
    false_dict = {}
    model_save_file = None
    model_save = {}
    test_classifiers = ['GBDT']
    num = ['1']
    classifiers = {
        'GBDT': gradient_boosting_classifier
    }
    is_binary_class = (len(np.unique(y_train)) == 2)
    for classifier, i in zip(test_classifiers, num):
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](X_train, y_train)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(X_test)

        probility = model.predict_proba(X_test)

        dic = {name + '_%s_score' % classifier: probility}
        sio.savemat(save_dir + name + '_' + classifier + '_score' + '.mat', dic)
        if model_save_file != None:
            model_save[classifier] = model
        accuracy = metrics.accuracy_score(y_test, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))
        accuracy_dict['GBDT'] = 100 * accuracy

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))


def classifier_xgboost(X_train, X_test, y_train, y_test, save_dir, tree_num, name):
    # global accuracy_dict
    print("******************* xgboost ********************")
    # start_time = time.time()
    clf = XGBClassifier(
        silent=0,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        nthread=4,# cpu 线程数 默认最大
        learning_rate=0.1,  # 如同学习率
        min_child_weight=5,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=5,  # 构建树的深度，越大越容易过拟合
        gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=0.8,  # 随机采样训练样本 训练实例的子采样比
        max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        colsample_bytree=0.9,  # 生成树时进行的列采样
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        reg_alpha=1,  # L1 正则项参数
        # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        objective='multi:softmax',  # 多分类的问题 指定学习任务和相应的学习目标，返回预测类别，不是概率
        num_class=2,  # 类别数，多分类与 multisoftmax 并用
        n_estimators=tree_num,  # 树的个数
        seed=0 # 随机种子
        # eval_metric= 'auc'
    )
    clf.fit(X_train, y_train, eval_metric='auc')
    y_true, y_pred = y_test, clf.predict(X_test)
    auc = metrics.accuracy_score(y_true, y_pred) * 100
    print("Accuracy : %.4g" % auc)
    # cost_time = time.time() - start_time
    print("xgboost success!", '\n', "cost time:", cost_time, "(s)......")
    probility = clf.predict_proba(X_test)
    accuracy_dict['xgboost_' + str(tree_num)] = metrics.accuracy_score(y_true, y_pred) * 100
    dic = {name + '_xgboost_score': probility}
    sio.savemat(save_dir + name + '_xgboost_score' + str(tree_num) + '.mat', dic)
    return auc


def classifier_SVM(X_train, X_test, y_train, y_test, save_dir, kernel_list):
    auc_list = []
    cost_time = []
    for kernel in kernel_list:
        print('******************* svm_%s ********************' % kernel)
        start_time = time.time()
        model = SVC(kernel=kernel, probability=True, random_state=2)
        model.fit(X_train, y_train)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(X_test)
        probility = model.predict_proba(X_test)
        dic = {'hog_%s_score' % "SVM": probility}

        sio.savemat(save_dir + "hog_SVM_score_" + kernel + '.mat', dic)

        accuracy = metrics.accuracy_score(y_test, predict)
        print('%s  accuracy: %.2f%%' % (kernel, (100 * accuracy)))
        auc_list.append(accuracy * 100)
        cost_time.append(time.time() - start_time)
    return auc_list, cost_time


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    for num in range(10):
        auc_rows = []
        accuracy_dict = {}
        # print(os.getcwd())
        """
        loading data
        """

        # num = 1
        dataset = 'ddsm'
        path = "../data/" + dataset + "/"  # the source of label train and test
        path1 = "../data/" + dataset + "/cca_feature_test/cca_feature_test0." + str(num) + "/"  # the source of features

        print(path1)

        sift_labels = pd.read_csv(path + "label.csv", header=None)
        train_index = pd.read_csv(path + "Train.csv", header=None)
        test_index = pd.read_csv(path + "Test.csv", header=None)

        y_train = sift_labels[train_index[0]].as_matrix().flatten()
        y_test = sift_labels[test_index[0]].as_matrix().flatten()

        name_list = ['SG_cca_sum.csv','SH_cca_sum.csv','SV_cca_sum.csv','GH_cca_sum.csv','GV_cca_sum.csv','HV_cca_sum.csv','SG_cca_concat.csv','SH_cca_concat.csv','SV_cca_concat.csv','GH_cca_concat.csv','GV_cca_concat.csv','HV_cca_concat.csv']
        # name_list = ['SG_cca_sum.csv', 'siftvgg16_cca_sum.csv', 'siftres50_cca_sum.csv', 'gistvgg16_cca_sum.csv', 'gistres50_cca_sum.csv',
        #              'vgg16res50_cca_sum.csv', 'SG_cca_concat.csv', 'siftvgg16_cca_concat.csv', 'siftres50_cca_concat.csv',
        #              'gistvgg16_cca_concat.csv', 'gistres50_cca_concat.csv', 'vgg16res50_cca_concat.csv']

        # for name in os.listdir(path1):
        for name in name_list:

            accuracy_dict['filename'] = name
            """
            saving path  
            """
            save_dir = "../data/result/" + dataset + "/" + name.split(".")[0] +"0."+ str(
                num) + '/'  # the folder to save accuracy mat
            save_csv_name = "../data/result/" + dataset + "/cca_feature_fuse" + "0."+ str(
                num) + '_record.csv'  # the path to save data record csv
            # save_csv_name = "../data/result/" + dataset + "/cca_feature_fuse_record.csv"
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            sift_features = pd.read_csv(path1 + name, header=None)

            print(sift_features.shape)

            X_train = sift_features[train_index[0]].as_matrix()
            X_test = sift_features[test_index[0]].as_matrix()

            num_train, num_feat = X_train.shape
            num_test, num_feat = X_test.shape
            print('\n\n******************** the feature of {} *********************'.format(name))
            print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))

            name = name.split(".")[0]

            """
            choose the classifier that you want to use
            """
            classifier_1(X_train, X_test, y_train, y_test, save_dir, name)
            classifier_NB(X_train, X_test, y_train, y_test, save_dir,
                          name)  # NB need all the feature descriptor must be non-negative
            classifer_adaboost(X_train, X_test, y_train, y_test, save_dir, name)
            classifier_GBDT(X_train, X_test, y_train, y_test, save_dir, name)

            """
                this is only for xgboost
                """
            #
            # """
            # tree_num = [20,40,60,80,100,125,150,175,200,300,400,500,1000,2000]
            # tree_num = [20, 40,550]
            tree_num = [550]
            cost_time = []
            for item in tree_num:
                auc = classifier_xgboost(X_train, X_test, y_train, y_test, save_dir, item, name)
                print('tree num is : %d \n' % item)

                # cost_time.append(time)
            print(accuracy_dict)
            auc_rows.append(accuracy_dict)
            print(auc_rows)
            accuracy_dict = {}
            print(accuracy_dict)

        print(auc_rows)
        headers = ['filename', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'NB', 'adaboost', 'GBDT', 'xgboost_20', 'xgboost_40']

        with open(save_csv_name, 'w') as f:
            f_csv = csv.DictWriter(f, headers)
            f_csv.writeheader()
            f_csv.writerows(auc_rows)

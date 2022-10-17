import operator
import random
from statistics import mean

import numpy as np
from skimage.io import imread
from sklearn.metrics import classification_report, confusion_matrix

from config import THRESHOLD_TRAIN_SIZE
from data_explorer import get_data_and_count
from utils import restructure_time_data_dict, plot_confusion_matrix


def split_train_test(data_dict, train_size):
    """
    Splits the data into test-train
    :param data_dict :(dictionary) data_dict --> {radiation:[img_brightness]}
    :param train_size : (float) Train size
    :return: train_dict-->{radiation:[img_brightness]} ,
             test_dict -->{radiation:[img_brightness]}
    """

    assert 0 < train_size < 1, "Invalid test size... I quit.."

    train_dict = {rad_label: [] for rad_label in data_dict}
    test_dict = {rad_label: [] for rad_label in data_dict}
    for rad_label, rad_file_list in data_dict.items():
        random.shuffle(rad_file_list)
        train_idx = int(train_size * (len(rad_file_list)))
        train_dict[rad_label] = rad_file_list[:train_idx]
        test_dict[rad_label] = rad_file_list[train_idx:]

    return train_dict, test_dict


def compute_traindata_mean(tr_dict):
    """
    Computes the mean of a class
    :param tr_dict : (dictionary) tr_dict --> {radiation_label:mean(avg img. brightness)}
    :return: (dict): {class_label: avg. class brightness}
    """
    mean_dict = {}
    for rad_label in tr_dict:
        mean_dict[rad_label] = mean(tr_dict[rad_label])

    return mean_dict


def test_classification(test_data, class_mean, time_period):
    """
    Finds the class of each test image based on their distance to the average class
    brightness
    :param test_data: (dictionary) : {class_label: [avg. img brightness]}
    :param class_mean: (dictionary) : {class_label: Avg. class brightness}
    :param time_period: (int/str): time interval at which the data was sampled
    :return: None
    """
    prediction_list = []
    # iteration through test dictionary
    for rad_label, mean_list in test_data.items():
        # iteration through each mean val in mean list
        for mean_val in mean_list:
            # dist_list --> [(label, distance)]
            dist_list = [(cls, abs(cls_mean - mean_val)) for cls, cls_mean in class_mean.items()]
            # finding the lowest distance tuple in dist_list
            prediction = min(dist_list, key=operator.itemgetter(1))
            # prediction_list --> [(class_label, predicted_label)]
            prediction_list.append((rad_label, prediction[0]))  # appending to pred
            # list, (original class label, predicted class label)

    y_test = [tup[0] for tup in prediction_list]
    y_pred = [tup[1] for tup in prediction_list]
    print("Accuracy on unknown data is", classification_report(y_test, y_pred))
    plot_confusion_matrix(
        confusion_matrix(y_test, y_pred), f'confusion_matrix_time' f'{time_period}.png'
    )


def classifier(data_dict, time_period=None):
    """
    Brightness based classifier
    :param data_dict: (dictionary): {radiation:[files]}
    :param time_period: (string/int): time interval at which the data was sampled
    :return: None
    """
    rad_list = [rad for rad in data_dict]
    brightness_dict = {rad_idx: [] for rad_idx, _ in enumerate(rad_list)}
    print(f"Classes used in classification: {rad_list}")

    for radiation, file_list in data_dict.items():
        for file in file_list:
            file_data = imread(file)
            avg_brightness = np.mean(file_data)
            brightness_dict[rad_list.index(radiation)].append(avg_brightness)

    train_data, test_data = split_train_test(brightness_dict, train_size=THRESHOLD_TRAIN_SIZE)
    mean_dict = compute_traindata_mean(train_data)
    print(f"Average class brightness: {mean_dict}")
    test_classification(test_data, mean_dict, time_period)


def time_based_classifier_wrapper(time_dict):
    """
    Wrapper to run the classier() on time based data
    :param time_dict: (dictionary) {time_period_of sampling:{radiation:{files}}}
    :return: None
    """
    restructured_data_dict = restructure_time_data_dict(time_dict)
    for time_period, data_dict in restructured_data_dict.items():
        print(f"Time period under consideration: {time_period} hours after radiation")
        classifier(data_dict, time_period=time_period)


def main():
    data, time_data = get_data_and_count()
    classifier(data)  # Classifier for the entire dataset
    time_based_classifier_wrapper(time_data)  # Classifier for the time based data


if __name__ == "__main__":
    main()

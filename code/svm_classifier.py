import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import DATA_BASE_PATH
from utils import get_data_and_count, compute_padding, restructure_time_data_dict


def data_processor(data_dict):
    """
    Creates padded input images and their labels
    :param data_dict: {dict} contains {radiation:[file paths]}
    :return: Pandas dataframe for data and label
    """
    data_list = []
    label_list = []
    rad_list = [rad for rad in data_dict]
    radiations_label_map = {radiation: rad_idx for rad_idx, radiation in enumerate(rad_list)}
    print(radiations_label_map)
    radiations_label_map = {f'{DATA_BASE_PATH}Fe':0, f'{DATA_BASE_PATH}X-ray':1}
    for radiation, file_list in data_dict.items():
        for file in file_list:
            img_data = imread(file)
            padding = compute_padding(img_data.shape)
            resized_img = np.pad(img_data, padding, mode='constant', constant_values= 0)
            scaled_img = MinMaxScaler(feature_range=(0, 1)).fit_transform(resized_img)
            data_list.append(scaled_img.flatten())
            label_list.append(radiations_label_map[radiation])
    data = np.array(data_list)
    label = np.array(label_list)
    df = pd.DataFrame(data)
    df['Target'] = label
    pd_data = df.iloc[:,:-1]
    pd_label = df.iloc[:,-1]
    return pd_data,pd_label


def svm_classifier(data, labels):
    """
    SVM training and testing
    :param data: radiation images
    :param labels: Labels for the images
    :return: None
    """
    svc = svm.SVC(kernel='linear',gamma='auto',probability=True)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30,
                                                        shuffle=True)
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)

    print(f"Classification report on unknown data : "
          f"{classification_report(y_test,y_pred)}")
    print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")

def main():
    _, time_data = get_data_and_count()
    restructured_data_dict = restructure_time_data_dict(time_data)
    for time_period, data_dict in restructured_data_dict.items():
        print(f"Time period under consideration: {time_period} hours after radiation")
        data, labels = data_processor(data_dict)
        svm_classifier(data, labels)


if __name__ == "__main__":
    main()

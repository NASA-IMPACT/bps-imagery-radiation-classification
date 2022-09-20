import numpy as np
import pandas as pd
from skimage.io import imread

# from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from thundersvm import SVC

from config import SVM_TEST_SIZE
from data_explorer import get_data_and_count, compute_padding, restructure_time_data_dict


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
    for radiation, file_list in data_dict.items():
        for file in file_list:
            img_data = imread(file)
            ascol = img_data.reshape(-1, 1)
            ascol_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(ascol)
            scaled_img = ascol_scaled.reshape(img_data.shape)
            padding = compute_padding(scaled_img.shape)
            resized_img = np.pad(scaled_img, padding, mode="constant", constant_values=0)
            data_list.append(resized_img.flatten())
            label_list.append(radiations_label_map[radiation])
    data = np.array(data_list)
    label = np.array(label_list)
    # visualize_data_scatter(data)
    df = pd.DataFrame(data)
    df["Target"] = label
    pd_data = df.iloc[:, :-1]
    pd_label = df.iloc[:, -1]

    return pd_data, pd_label


def svm_classifier(data, labels):
    """
    SVM training and testing
    :param data: radiation images
    :param labels: Labels for the images
    :return: None
    """
    svc = SVC(kernel="linear", gamma="auto", probability=True)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=SVM_TEST_SIZE, shuffle=True
    )
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)

    print(f"Classification report on unknown data : " f"{classification_report(y_test,y_pred)}")
    print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")


def main():
    _, time_data = get_data_and_count()
    restructured_data_dict = restructure_time_data_dict(time_data)
    time = input("Enter the data to be used in the classifier (Valid options: 4,24,48): ")
    if int(time) not in [4, 24, 48]:
        raise ValueError('Invalid option entered !!, I quit !!')
    data, labels = data_processor(restructured_data_dict[int(time)])
    svm_classifier(data, labels)


if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerGradCam, LayerAttribution
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from BPSDataLoader import load_torch_data
from Model import RadNet
from config import (
    BATCH_SIZE,
    LR,
    ADAM_B2,
    ADAM_B1,
    EPOCHS,
    CNN_ACCU_PLOT,
    CNN_LOSS_PLOT,
    CNN_PATH,
    N_COLS,
    N_ROWS,
)
from ptorchtools import EarlyStopping
from utils import get_data_and_count, restructure_time_data_dict
from utils import plot_data, compute_correct_pred_torch, plot_confusion_matrix, visualize_gradcam

device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

earlystopping = EarlyStopping(patience=5, verbose=True)


def train_cnn(traindata, valdata, model_params):
    """
    CNN training
    :param traindata: Dataset object of training data
    :param valdata: Dataset object of validation data
    :param model_params: (dict) {param:value}
    :return:  None
    """
    train_loader = DataLoader(traindata, batch_size=BATCH_SIZE)
    val_loader = DataLoader(valdata, batch_size=BATCH_SIZE)
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    cnn = model_params['model']
    cnn.to(device)
    optimizer = model_params['optimizer']
    loss_criterion = model_params['loss']
    for epoch in range(EPOCHS):
        running_tr_loss = 0.0
        running_tr_corr = 0.0
        tr_total = 0.0
        for tr_idx, data in enumerate(train_loader):
            train_img, img_label = data[0].to(device), data[1].to(device)
            cnn.train()
            optimizer.zero_grad()
            train_output = cnn(train_img)
            tr_loss = loss_criterion(train_output, img_label)
            tr_total += img_label.shape[0]
            running_tr_corr += compute_correct_pred_torch(train_output, img_label)
            tr_loss.backward()
            optimizer.step()
            running_tr_loss += tr_loss.item()
        train_loss.append(running_tr_loss / len(train_loader))
        train_accuracy.append(running_tr_corr / tr_total)

        with torch.no_grad():
            running_val_loss = 0.0
            running_val_corr = 0.0
            val_total = 0.0
            cnn.eval()
            for val_idx, val_data in enumerate(val_loader):
                val_img, val_label = val_data[0].to(device), val_data[1].to(device)
                val_output = cnn(val_img)
                val_loss = loss_criterion(val_output, val_label)
                val_total += val_label.shape[0]
                running_val_loss += val_loss.item()
                running_val_corr += compute_correct_pred_torch(val_output, val_label)
        valid_loss.append(running_val_loss / len(val_loader))
        valid_accuracy.append(running_val_corr / val_total)
        print(
            f"Epoch : {epoch}, train loss:{running_tr_loss / len(train_loader)}, "
            f"valid loss: {running_val_loss / len(val_loader)}, "
            f"train accuracy: {running_tr_corr / tr_total}, "
            f"validation accuracy: {running_val_corr / val_total}"
        )

        earlystopping(np.average(running_val_loss / len(val_loader)), cnn)
        if earlystopping.early_stop:
            print(" Early Stopping !!!")
            break

    plot_data(train_loss, valid_loss, CNN_LOSS_PLOT)
    plot_data(train_accuracy, valid_accuracy, CNN_ACCU_PLOT)


def test_cnn(testdata, model_params, grad_cam=False):
    """
    CNN testing
    :param testdata:Dataset object of testing data
    :param model_params: (dict) {param:value}
    :return:
    """
    test_loader = DataLoader(testdata, batch_size=BATCH_SIZE)
    correct = 0
    total = 0
    all_pred = []
    all_labels = []
    cnn = model_params['model']
    cnn.to(device)
    cnn.load_state_dict(torch.load(CNN_PATH))

    gradcam = LayerGradCam(cnn, cnn.conv2)

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = cnn(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            _, act_label = torch.max(labels.data, 1)

            if grad_cam:
                attribution = gradcam.attribute(images, target=act_label)
                upsampled_attr = LayerAttribution.interpolate(attribution, (N_ROWS, N_COLS))
                visualize_gradcam(upsampled_attr, images, idx, act_label, predicted)

            all_pred.extend([i for i in predicted.cpu().numpy()])
            all_labels.extend([i for i in act_label.cpu().numpy()])
            total += labels.size(0)
            correct += compute_correct_pred_torch(outputs, labels)

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    plot_confusion_matrix(confusion_matrix(all_labels, all_pred))
    print(classification_report(all_labels, all_pred))


def train_test_wrapper(data_path_dict, test_only=False):
    """
    Wrapper for train-test
    :param data_path_dict: (dict) {radiation:[file_paths]}
    :return: None
    """

    model_params = {}
    train_data, val_data, test_data = load_torch_data(data_path_dict)
    cnn = RadNet()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
    loss_criterion = nn.CrossEntropyLoss()
    model_params['optimizer'] = optimizer
    model_params['loss'] = loss_criterion
    model_params['model'] = cnn
    if not test_only:
        train_cnn(train_data, val_data, model_params)
    test_cnn(test_data, model_params, grad_cam=False)


def main():
    data_paths, time_data_paths = get_data_and_count()
    time_data_dict = restructure_time_data_dict(time_data_paths)
    # train_test_wrapper(data_paths) # use this running the classifier on the entire data
    time = input("Enter the data to be used in the classifier (Valid options: 4,24,48): ")
    if int(time) not in [4, 24, 48]:
        raise ValueError('Invalid option entered !!, I quit !!')
    # set test_only to True when only testing the network (ie. disable training)
    train_test_wrapper(time_data_dict[int(time)], test_only=False)


if __name__ == "__main__":
    main()

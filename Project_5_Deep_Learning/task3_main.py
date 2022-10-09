"""
    CS5330 - Computer Vision - Bruce Maxwell
    Project: Recognition using Deep Networks
    Names: Sida Zhang & Hongyu Wan

    Description: Create greek letter embedding space
    and compute the distance in the embedding space
"""
# import statements
from torch.utils.data import Dataset
from task1_main import *
import csv
import random
import math
import os
from PIL import Image
import numpy as np

# class definitions: define new sub network model from Net
class Greek_Submodel(Net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward(self, x):
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # A dropout layer with a 0.5 dropout rate (50%)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        # ReLU function,fully connected Linear layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x)


# Process the test data
def process_example(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    return batch_idx, example_data, example_targets

# create greek letter dataset
def create_dataset(path, dataset_filepath, label_filepath):
    files = os.listdir(path)
    with open(dataset_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset'])
    for file in files:
        position = path + '\\' + file
        print(position)
        img = Image.open(position)
        img = img.resize((28, 28), Image.NEAREST)
        #greyscaled
        img = img.convert('LA')

        img = np.array(img)
        img = img[:, :, 0]
        #threshold to binary image because input image and dataset image are not very similar
        for x in range(28):
            for y in range(28):
                if (img[x][y] > 100):
                    img[x][y] = 255
                if (img[x][y] < 100):
                    img[x][y] = 0
        with open(dataset_filepath, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(img.flatten())
    with open(label_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label(alpha = 0, beta = 1, gamma = 2)'])
        for file in files:
            #         position = path+'\\'+ file
            print(file)
            if "alpha" in file: writer.writerow('0')
            if "beta" in file: writer.writerow('1')
            if "gamma" in file: writer.writerow('2')
    return

# read dataset from csv
def read_dataset(dataset_filepath, label_filepath):
    # read greek dataset（27*28*28）
    label = []
    with open(dataset_filepath, "r", newline='') as f:
        head_row = next(f)
        reader = csv.reader(f)
        data = list(reader)
        data = np.array(data)
        dataset = []
        for line in data:
            arr_2d = np.reshape(line, (28, 28))
            dataset.append(arr_2d)

    dataset = np.array(dataset)
    # read label
    with open(label_filepath, "r", newline='') as f:
        head_row = next(f)
        reader = csv.reader(f)
        # label = list(reader)
        for row in reader:
            label.append(row[0])
        label = np.array(label)
    return dataset, label


# random choose greek m kinds from n greek_symbols
def rand_choose_greek(n, m):
    rand_index = []
    for i in range(m):
        rand_index.append(random.randint(n / m * i, n / m * (i + 1) - 1))
    return rand_index

# compute sum squared distance of two tensor values
def compute_ssd(indexs, tensor2array, label, greek_symbol):
    value_all = []
    for index in indexs:
        print("We have randomly choose an image from the dataset: ", index)
        # for i in range(len(tensor2array)):
        # value_index=tensor2array[0,index+1]
        for i in range(len(tensor2array)):
            value = math.sqrt(sum((tensor2array[index] - tensor2array[i]) * (tensor2array[index] - tensor2array[i])))
            if value == 0:
                value_all.append(i)
            print("SSD of image (", index, ") \t-\t", greek_symbol[int(label[i])], " \t-\t", round(value, 2))
    for i in range(len(value_all)):
        print("Best Match Image after Sum Squared Distance (", int(value_all[i]), ") is :",
              greek_symbol[int(label[int(value_all[i])])])


# get new_greekdata from file
def get_new_greekdata(path):
    files = os.listdir(path)
    input_image = []
    raw_image=[]
    for file in files:
        position = path + '\\' + file
        print(position)
        img = Image.open(position)
        img = img.resize((28, 28), Image.NEAREST)
        raw_img = img
        raw_img = np.array(raw_img)
        raw_image.append(raw_img)
        img = img.convert('LA')
        img = np.array(img)
        img = img[:, :, 0]
        #threshold to binary image because input image and dataset image are not very similar
        for x in range(28):
            for y in range(28):
                if (img[x][y] > 100):
                    img[x][y] = 255
                if (img[x][y] < 100):
                    img[x][y] = 0
        input_image.append(img)
    return input_image, raw_image

# datatype convert
def change_datatype(input_image):
    input_image = np.array(input_image)
    input_image_x = input_image.astype(float)
    input_image_t = torch.tensor(input_image_x)
    input_image_t = input_image_t.unsqueeze(1)
    input_image_y = torch.tensor(input_image_t, dtype=torch.float)
    return input_image_y

# return the minimum value from a list
def get_minvalue(input_list):
    min_value = min(input_list)

    # return the index of minimum value
    min_index = input_list.index(min_value)
    return min_index

# compute the distance of untrained images
def compute_ssd_untrained(tensor2array, new_output):
    value_all = []
    # for i in range(len(tensor2array)):
    # value_index=tensor2array[0,index+1]
    for i in range(len(tensor2array)):
        value = math.sqrt(sum((new_output - tensor2array[i]) * (new_output - tensor2array[i])))
        value_all.append(value)
    min_index = get_minvalue(value_all)
    return min_index


# defined different hyperparameters, run network, and return results
def main(argv):
    # define the learning rate and momentum hyperparameters for optimizer
    # define three greek symbols
    greek_symbol = ['alpha', 'beta', 'gamma']
    batch_size_train = 10
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)
    # A. Create a greek symbol data set
    create_dataset('../data/greek_data', '../results/greek_symbol_csv/dataset.csv', '../results/greek_symbol_csv/label.csv')

    # B. Create a truncated model
    Greek_submodel = Greek_Submodel()
    Greek_submodel_optimizer = optim.SGD(Greek_submodel.parameters(), lr=learning_rate,
                                         momentum=momentum)
    # load model
    network_state_dict = torch.load('../results/networkTrained/model.pth')
    Greek_submodel.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('../results/networkTrained/optimizer.pth')
    Greek_submodel_optimizer.load_state_dict(optimizer_state_dict)

    # load mnist dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    batch_idx, example_data, example_targets = process_example(train_loader)
    Greek_submodel.eval()
    # C. Project the greek symbols into the embedding space
    with torch.no_grad():
        # first training input image
        output = Greek_submodel(example_data[0])
        print("The element vectors has size of: ", output.shape)

    # D. Project the greek symbols into the embedding space
    dataset, label = read_dataset('../results/greek_symbol_csv/dataset.csv', '../results/greek_symbol_csv/label.csv')
    dataset_y = change_datatype(dataset)
    Greek_submodel.eval()
    with torch.no_grad():
        output_greek = Greek_submodel(dataset_y)

    # rand choose m kinds of greek_symbol from n symbol
    index = rand_choose_greek(len(label), 3)
    print(index)
    output_array = output_greek.numpy()
    compute_ssd(index, output_array, label, greek_symbol)

    # E. Create your own greek symbol data
    input_image,raw_img = get_new_greekdata('../data/greek_input')
    # print((np.array(input_image)).shape)
    input_image_y = change_datatype(input_image)
    Greek_submodel.eval()
    with torch.no_grad():
        output_new = Greek_submodel(input_image_y)

    # plot the prediction
    for pic in range(len(output_new)):
        min_index = compute_ssd_untrained(output_array, output_new[pic])
        plt.subplot(3, 3, pic + 1)
        plt.tight_layout()
        plt.imshow(raw_img[pic], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(greek_symbol[int(label[min_index])]))
        plt.xticks([])
        plt.yticks([])
    plt.show()



if __name__ == "__main__":
    main(sys.argv)

"""
    CS5330 - Computer Vision - Bruce Maxwell
    Project: Recognition using Deep Networks
    Names: Sida Zhang & Hongyu Wan

    Description: For this program, we have wrote our on database
    with three new greek letters "Pi, Theta, and Mu"
    Each of the greek letter have 11 training sets and 2 test sets
    We have built an actual KNN classifier that can take in any
    square image and classify it.
"""

# import statements
from task1_main import *
import csv
import math
from collections import Counter


# class definitions: define a customized net work network model with two convolutional layers
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

# created dataset for 6 labels in a csv file
def create_dataset(path, dataset_filepath, label_filepath):
    files = os.listdir(path)
    with open(dataset_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset'])
    for file in files:
        position = path + '\\' + file
        img = Image.open(position)
        img = img.resize((28, 28), Image.NEAREST)
        #greyscaled
        img = img.convert('LA')
        img = np.array(img)
        img = img[:, :, 0]
        #threshold to binary image because input image and dataset image are not very similar
        for x in range(28):
            for y in range(28):
                if (img[x][y] > 150):
                    img[x][y] = 255
                else:
                    img[x][y] = 0
        with open(dataset_filepath, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(img.flatten())
    with open(label_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label(alpha = 0, beta = 1, gamma = 2, pi = 3, theta = 4, mu = 5)'])
        for file in files:
            #         position = path+'\\'+ file
            if "alpha" in file: writer.writerow('0')
            if "beta" in file: writer.writerow('1')
            if "gamma" in file: writer.writerow('2')
            if 'pi' in file: writer.writerow('3')
            if 'theta' in file: writer.writerow('4')
            if 'mu' in file: writer.writerow('5')
    return

# read dataset from the csv file
def read_dataset(dataset_filepath, label_filepath):
    # read greek dataset（27*28*28）
    label = []
    dataset = []
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


# get new_greekdata from file
def get_test_greekdata(path):
    files = os.listdir(path)
    input_image = []
    raw_image = []
    for file in files:
        position = path + '\\' + file
        img = Image.open(position)
        img = img.resize((28, 28), Image.NEAREST)
        raw_img = img
        raw_img = np.array(raw_img)
        raw_image.append(raw_img)
        #greyscaled
        img = img.convert('LA')
        img = np.array(img)
        img = img[:, :, 0]
        #threshold to binary image because input image and dataset image are not very similar
        for x in range(28):
            for y in range(28):
                if (img[x][y] > 150):
                    img[x][y] = 255
                else:
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

# get the index of knn value where k is 4 set to 4 in our program
# return the index of the set with the most frequency
def get_knnvalue(distance, label):
    smallest_three = []
    knn_label = []
    distance_list = distance.copy()
    distance_list.sort()
    for i in range(4):
        smallest_three.append(distance_list[i])
        print(distance.index(distance_list[i]))
        knn_label.append(label[distance.index(distance_list[i])])
        # print(smallest_three)
    # Counter returns the frequency
    collection_num = Counter(knn_label)
    most_counterNum = collection_num.most_common(1)
    most_counterNum = np.array(most_counterNum)
    return most_counterNum[0][0]

# compute the sum squared distance and compute the knn value
def compute_knn_testset(tensor2array, new_output, label):
    value_all = []
    for i in range(len(tensor2array)):
        value = math.sqrt(sum((new_output - tensor2array[i]) * (new_output - tensor2array[i])))
        value_all.append(value)
    min_index = get_knnvalue(value_all, label)
    return min_index


# defined different hyperparameters, run network, and return results
def main(argv):
    # define the learning rate and momentum hyperparameters for optimizer
    # define six greek symbols
    greek_symbol = ['alpha', 'beta', 'gamma', 'pi', 'theta', 'mu']
    learning_rate = 0.01
    momentum = 0.5

    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)
    # A. Create a greek symbol data set
    create_dataset('../extension/extension_train', '../results/extension_csv/dataset.csv', '../results/extension_csv/label.csv')

    # B. Create a truncated model
    Greek_submodel = Greek_Submodel()
    Greek_submodel_optimizer = optim.SGD(Greek_submodel.parameters(), lr=learning_rate,
                                         momentum=momentum)
    # load model
    network_state_dict = torch.load('../results/networkTrained/model.pth')
    Greek_submodel.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('../results/networkTrained/optimizer.pth')
    Greek_submodel_optimizer.load_state_dict(optimizer_state_dict)

    # D. Project the greek symbols into the embedding space
    dataset, label = read_dataset('../results/extension_csv/dataset.csv', '../results/extension_csv/label.csv')
    dataset_y = change_datatype(dataset)
    Greek_submodel.eval()
    with torch.no_grad():
        output_greek = Greek_submodel(dataset_y)
    output_array = output_greek.numpy()

    # E. Get test dataset
    input_image, raw_img = get_test_greekdata('../extension/extension_test')
    input_image_y = change_datatype(input_image)
    Greek_submodel.eval()
    with torch.no_grad():
        output_new = Greek_submodel(input_image_y)

    # KNN and plot the prediction
    for pic in range(len(output_new)):
        min_index = compute_knn_testset(output_array, output_new[pic], label)
        plt.subplot(3, 4, pic + 1)
        plt.tight_layout()
        plt.imshow(raw_img[pic], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(greek_symbol[int(min_index)]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main(sys.argv)

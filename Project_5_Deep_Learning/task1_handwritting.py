"""
    CS5330 - Computer Vision - Bruce Maxwell
    Project: Recognition using Deep Networks
    Names: Sida Zhang & Hongyu Wan

    Description: Created Hand-writting data sets and
     test the network on new Hand-writtings
"""
# import statements

import os
from torch.utils.data import DataLoader
from task1_main import *
from PIL import Image
import numpy as np

# new class definitions: define new class handwritting dataset and parse the data
class HanddSet(data.Dataset):
    def __init__(self, root):
        # path
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize([28, 28]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,)),

        ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = torchvision.transforms.functional.invert(pil_img)
            data = self.transforms(data)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)


# Process the test data
def pro_example(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data) = next(examples)
    return batch_idx, example_data


# defined different hyperparameters, run network, and return results
def main(argv):
    # define how many times we run and the test and train set size.
    # define the learning rate and momentum hyperparameters for optimizer
    learning_rate = 0.01
    momentum = 0.5
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)

    # load dataset from /data from downloaded files' location
    dataSet = HanddSet('../data/handwritting/')
    test_loader = DataLoader(dataset=dataSet,  # dataset
                             batch_size=9,  # batch size
                             shuffle=True,  # shuffle
                             num_workers=0)
    batch_idx, example_data = pro_example(test_loader)

    # main function code
    # load trained models
    continued_network = Net()
    continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate,
                                    momentum=momentum)
    # load mnist dataset from downloaded files' location
    network_state_dict = torch.load('../results/networkTrained/model.pth')
    continued_network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load('../results/networkTrained/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)
    continued_network.eval()
    with torch.no_grad():
        output = continued_network(example_data)
        for i in range(9):
            predict = output.data.max(1, keepdim=True)[1][i]
    plt_test_dataset(example_data, output)
    return


if __name__ == "__main__":
    main(sys.argv)

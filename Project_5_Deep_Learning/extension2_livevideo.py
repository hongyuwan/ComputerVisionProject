"""
    CS5330 - Computer Vision - Bruce Maxwell
    Project: Recognition using Deep Networks
    Names: Sida Zhang & Hongyu Wan

    Description: A live video digit recognition application
    using the trained network.
"""
# import statements
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from task1_main import *
from torchvision import transforms
from task1_main import *
import cv2


# datatype convert
def change_datatype(input_image):
    input_image = np.array(input_image)
    input_image_x = input_image.astype(float)
    input_image_t = torch.tensor(input_image_x)
    input_image_t = input_image_t.unsqueeze(1)
    input_image_y = torch.tensor(input_image_t, dtype=torch.float)
    return input_image_y


def pro_example(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data) = next(examples)
    return batch_idx, example_data

# defined different hyperparameters, run network, and return results in console
def main(argv):
    # handle any command line arguments in argv
    batch_size_train = 64
    batch_size_test = 10
    learning_rate = 0.01
    momentum = 0.5
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)
    # interval when printing results, set to print result every 25 frames
    interval = 0

    continued_network = Net()
    continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate,
                                    momentum=momentum)
    network_state_dict = torch.load('../results/networkTrained/model.pth')
    continued_network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load('../results/networkTrained/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)
    cap = cv2.VideoCapture(0)
    while 1:
        interval += 1
        ret, frame = cap.read()
        # show the original frame
        frame = cv2.resize(frame, (280, 280))
        cv2.imshow("source", frame)
        # show the grayscaled and binary image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 灰度化
        res, frame = cv2.threshold(frame, 90, 255, cv2.THRESH_BINARY_INV)  # 反向二值化
        cv2.imshow("gray", frame)
        # show the 28x28 image
        frame = cv2.resize(frame, (140, 140))
        cv2.imshow("28*28", frame)
        cv2.waitKey(100)

        testimg = cv2.resize(frame, (28, 28))
        # batch_idx, example_data = pro_example(testimg)
        # resize the image size to 28x28 size
        testimg = torch.Tensor(testimg)
        testimg = torch.unsqueeze(testimg, dim=0)
        testimg = torch.unsqueeze(testimg, dim=0)
        # testimg = change_datatype(testimg)
        continued_network.eval()
        with torch.no_grad():
            output = continued_network(testimg)
            predict = output.data.max(1, keepdim=True)[1]
            # interval when printing results, set to print result every 25 frames
            if interval % 25 == 0:
                print("The Prediction of the input image is: \t", predict[0][0].item())


if __name__ == "__main__":
    main(sys.argv)
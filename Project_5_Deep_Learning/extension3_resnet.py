"""
    CS5330 - Computer Vision - Bruce Maxwell
    Project: Recognition using Deep Networks
    Names: Sida Zhang & Hongyu Wan

    Description: This progoram downloads ResNetx from pytorch
    resnext50_32x4d and run the submodel(torchvision)
"""

# import statements
from task1_main import *
from torchvision import transforms
import cv2
from torchvision import models

# class definitions: defined sub network model from ResBet from torchvision
class Submodel(models.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = F.relu( F.max_pool2d( self.conv1(x), 2 ) ) # relu on max pooled results of conv1
        x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) ) # relu on max pooled results of dropout of conv2
        return x

# this function plot the convolutional layers after 5x5 filter
def plot_conv_filters(model_weights, n, m):
    plt.figure()
    for i, filter_layer in enumerate(model_weights):
        plt.subplot(n, m, i + 1)  # we have 5x5 filters and total of 16
        plt.imshow(filter_layer[0, :, :].detach().cpu().numpy(), cmap='gist_gray')
        plt.axis('off')
    plt.show()

# analyze the weights from the convolutional layers
def analysis(continued_network):
    model_weights = []
    conv_layers = []
    model_children = list(continued_network.children())
    print(model_children)
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    return model_weights, conv_layers

# apply filters to the first image by using cv2
def applyFiltersToImage(filters, image):
    print(filters.shape)
    filteredImgs = []
    for filterNum, filter_layer in enumerate(filters):
        filter = filters[filterNum, 0, :, :]
        dst = cv2.filter2D(image, -1, filter)
        # -1 to use src depth
        # if min val is negative, shift values to be in non-negative range
        minVal = dst.min()
        if minVal < 0:
            dst = dst + -minVal
            maxVal = dst.max()
            dst = dst * 255.0 / maxVal
        # img rows x cols, 3 color channels
        filteredWColor = np.zeros((dst.shape[1], dst.shape[2], 3))
        filteredWColor[:, :, 0] = dst  # red
        filteredWColor[:, :, 1] = dst  # green
        filteredWColor[:, :, 2] = dst  # blue
        filteredWColor = filteredWColor.astype("uint8")
        filteredImgs.append(filteredWColor)
    # plot the result in plt
    plt.figure()
    for filterNum, filter_layer in enumerate(filters):
        # we have 5x5 filters
        plt.subplot(12, 12, filterNum * 2 + 1)
        plt.imshow(filter_layer[0, :, :], cmap='gist_gray')
        plt.axis('off')
        plt.subplot(12, 12, filterNum * 2 + 2)
        plt.axis('off')
        plt.imshow(filteredImgs[filterNum])
    plt.show()

# apply filters to the after-effect image by using cv2
def applyFiltersToImage_first(filters, image):
    print(filters.shape)
    filteredImgs = []
    for filterNum, filter_layer in enumerate(filters):
        print(filters.shape)
        filter = filters[filterNum ,0, :, : ]
        dst = cv2.filter2D(image, -1, filter)  # -1 to use src depth
        print(dst.shape)
        # if min val is negative, shift values to be in non-negative range

        filteredImgs.append(dst)
        final_image=dst
    return final_image

# plot both output and after-effect images by using cv2
def applyFiltersToImage_withresult(filters, image,output):
    filteredImgs = []
    for filterNum, filter_layer in enumerate(filters):
        filter = filters[filterNum ,0, :, : ]
        dst = cv2.filter2D(image, -1, filter)
        # -1 to use src depth
        # if min val is negative, shift values to be in non-negative range
        minVal = dst.min()
        if minVal < 0:
            dst = dst + -minVal
            maxVal = dst.max()
            dst = dst * 255.0 / maxVal

        filteredWColor = np.zeros((dst.shape[1], dst.shape[2], 3))
        # img rows x cols, 3 color channels
#         print(filteredWColor.shape)
#         print(filteredWColor)
        filteredWColor[:, :, 0] = dst  # red
        filteredWColor[:, :, 1] = dst  # green
        filteredWColor[:, :, 2] = dst  # blue

        filteredWColor = filteredWColor.astype("uint8")

        filteredImgs.append(filteredWColor)
    plt.figure()
    for filterNum, filter_layer in enumerate(filters):
        plt.subplot(5, 8, filterNum*2 + 1)
        plt.imshow(filteredImgs[filterNum])
        plt.axis('off')
        plt.subplot(5, 8, filterNum*2 + 2)
        plt.imshow(output[filterNum, :, :], cmap='gist_gray')
        plt.axis('off')
    plt.show()

# defined different hyperparameters, run network, and return results
def main(argv):
    # define the test and train set size.
    # define the learning rate and momentum hyperparameters for optimizer
    batch_size_train = 10
    batch_size_test = 10
    learning_rate = 0.01
    momentum = 0.5
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)
    # load trained model and optimizer
    continued_network = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)

    continued_network.eval()
    # A.Analyze the first layer
    model_weights, conv_layers = analysis(continued_network)
    plot_conv_filters(model_weights[0], 12, 12)
    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
    # Get Input IMage
    train_loader, test_loader = dataset_load(batch_size_train, batch_size_test)
    batch_idx, example_data, example_targets = process_example(train_loader)
    weight = model_weights[0].cpu().detach().numpy()
    data_plot = example_data.numpy()

    # B. Show the effect of the filters
    with torch.no_grad():
        applyFiltersToImage(weight, data_plot[0])

    # C. Build a truncated model
    net_submodel = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    net_submodel.eval()
    model_weights, conv_layers = analysis(net_submodel)
    # plot_conv_filters(model_weights[0], 5, 5)
    # plot_conv_filters(model_weights[1], 5, 5)
    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

    return
if __name__ == "__main__":
    main(sys.argv)

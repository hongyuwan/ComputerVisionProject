"""
    CS5330 - Computer Vision - Bruce Maxwell
    Project: Recognition using Deep Networks
    Names: Sida Zhang & Hongyu Wan

    Description: Read the network from task1_main and
        run it on the test set.
"""

# import functions from task1_main
from task1_main import *

# defined different hyperparameters from task1_main, run network, and return results
def main(argv):
    # define how many times we run and the test and train set size.
    # define the learning rate and momentum hyperparameters for optimizer
    batch_size_train = 64
    batch_size_test = 10
    learning_rate = 0.01
    momentum = 0.5
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)

    # load mnist dataset from downloaded files' location
    train_loader, test_loader = dataset_load(batch_size_train, batch_size_test)

    # process and parse data
    batch_idx, example_data, example_targets = process_example(test_loader)
    plt_dataset(example_data, example_targets)

    # main function code
    # load trained models
    continued_network = Net()
    continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate,
                                    momentum=momentum)
    network_state_dict = torch.load('../results/networkTrained/model.pth')
    continued_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('../results/networkTrained/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)
    continued_network.eval()
    with torch.no_grad():
        output = continued_network(example_data)
    plt_test_dataset(example_data, output)
    # print results in console
    for i in range(10):
        for j in range(10):
            if j == 9:
                print(j, ' = ', round(output.data[i][j].item(), 2), ',\t Pred. = \t',
                      output.data.max(1, keepdim=True)[1][i].item(), ', Actual = ',
                      example_targets[i].item())
            else:
                print(j, ' = ', round(output.data[i][j].item(), 2), ', ', end='\t')
    return


if __name__ == "__main__":
    main(sys.argv)
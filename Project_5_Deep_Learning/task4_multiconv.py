"""
    CS5330 - Computer Vision - Bruce Maxwell
    Project: Recognition using Deep Networks
    Names: Sida Zhang & Hongyu Wan

    Description: We have designed our new train with
    3 convolutional layers for this program.
"""

# import statements
from task1_main import *

# class definitions: define a customized net work network model with 3 convolutional layers
class Multiconv_Net(nn.Module):
    def __init__(self):
        super(Multiconv_Net, self).__init__()
        # A convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding='same')
        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5, padding='same')
        # A dropout layer with a 0.5 dropout rate (50%)
        self.conv3_drop = nn.Dropout2d(p=0.5)
        # fully connected Linear with 50
        self.fc1 = nn.Linear(270, 50)
        # fully connected Linear with 10
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # A dropout layer with a 0.5 dropout rate (50%)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 270)
        # ReLU function,fully connected Linear layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # final fully connected Linear layer with 10 nodes
        x = self.fc2(x)
        # log_softmax function
        return F.log_softmax(x)


# Train the model with different hyperparameters
def train_d(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            str1 = str('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            with open('../results/conv_original.txt', 'a') as f:
                f.writelines(str1)
                f.writelines('\n')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), '../results/networkTrained/model.pth')
            torch.save(optimizer.state_dict(), '../results/networkTrained/optimizer.pth')

# Test the model with different hyperparameters and return the avg loss and corectness
# and write to txt file in ../results/conv_original.txt
def test_d(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    str1 = str('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    with open('../results/conv_original.txt', 'a') as f:
        f.writelines(str1)
        f.writelines('\n')
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# defined hyperparameters and run network with 3 convolutional layers
def main(argv):
    # define all the hyperparameters
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)
    # overwrite the txt file everytime when we run this program with header row ''
    with open('../results/conv_original.txt', 'w') as f:
        f.writelines('')
    train_loader, test_loader = dataset_load(batch_size_train, batch_size_test)

    batch_idx, example_data, example_targets = process_example(test_loader)
    print(example_data.shape)
    plt_dataset(example_data, example_targets)

    # main function code and run the network
    network = Multiconv_Net()
    optimizer = optim.SGD(network.parameters(), lr= learning_rate, momentum= momentum)

    # initialize the train and test arrays
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    # initialize the first run test on downloaded dataset
    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        # run train on downloaded dataset
        train_d(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval)
        # run test on downloaded dataset
        test_d(network, test_loader, test_losses)
    print_loss(train_counter, train_losses, test_counter, test_losses)
    return


if __name__ == "__main__":
    main(sys.argv)

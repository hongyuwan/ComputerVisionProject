"""
    CS5330 - Computer Vision - Bruce Maxwell
    Project: Recognition using Deep Networks
    Names: Sida Zhang & Hongyu Wan

    Description: We have designed our new train with
    batch_size_list of [64, 128, 256],
    epochs_list of [1, 5, 10],
    learning_rate_list of [0.001, 0.01, 0.1], and
    dropout_rate_list of [0.125, 0.25, 0.5],
    for this program..
"""

# import statements
from task1_main import *
import csv



# class definitions: define a customized net work network model with customized dropout rate
class CustomizedNet(nn.Module):
    def __init__(self,arg):
        super(CustomizedNet, self).__init__()
        # A convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # A dropout layer with a 0.5 dropout rate (customized)
        self.conv2_drop = nn.Dropout2d(p=arg)
        # fully connected Linear with 50
        self.fc1 = nn.Linear(320, 50)
        # fully connected Linear with 10
        self.fc2 = nn.Linear(50, 10)

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
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), '../results/networkTrained/model.pth')
            torch.save(optimizer.state_dict(), '../results/networkTrained/optimizer.pth')

# Test the model with different hyperparameters and return the avg loss and corectness
def test_d(network, test_loader, test_losses, bool):
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
    if bool:
        # str1 = str(str(correct.cpu().detach().numpy().tolist()), '/', str(len(test_loader.dataset)))
        # str2 = str('({:.0f}%)'.format(100. * correct / len(test_loader.dataset)))
        correctness = correct.cpu().detach().numpy().tolist()
        percent = 100. * correctness / len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss, correctness, percent

# simply run this function 81 times because we have 3x3x3x3 options
def loop_exprienment(batch_size_train, n_epochs, learning_rate, dropout_rate):
    batch_size_test = 1000
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)

    train_loader, test_loader = dataset_load(batch_size_train, batch_size_test)

    batch_idx, example_data, example_targets = process_example(test_loader)
    print(example_data.shape)
    plt_dataset(example_data, example_targets)
    # main function code
    network = CustomizedNet(dropout_rate)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    test_d(network, test_loader, test_losses, False)

    for epoch in range(1, n_epochs + 1):
        train_d(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval)
        avg_loss, accuracy, percent = test_d(network, test_loader, test_losses, True)
    # print(len(test_counter))
    # print(len(test_losses))
    print_loss(train_counter, train_losses, test_counter, test_losses)

    result = [batch_size_train, n_epochs, learning_rate, dropout_rate, avg_loss, accuracy, percent]

    return result

# write batch_size_train, n_epochs, learning_rate, dropout_rate, avg_loss, accuracy, percent to csv
def csv_result_save(result):
    with open('../results/customized_result.csv', "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)

# defined different hyperparameters (3x3x3x3):
# the function runs about 2 hours with 81 variables
def main(argv):
    # define all the hyperparameters
    batch_size_list = [64, 128, 256]
    epochs_list = [1, 5, 10]
    learning_rate_list = [0.001, 0.01, 0.1]
    dropout_rate_list = [0.125, 0.25, 0.5]

    # first time overwrite csv with a header row
    with open('../results/customized_result.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Batch_Size', 'Epochs', 'Dropout_Rate', 'Learning_Rate', 'Avg.loss', 'Correctness',
             'Percent'])

    # loop all the hyperparameters
    for batch_size_train in batch_size_list:
        for n_epochs in epochs_list:
            for learning_rate in learning_rate_list:
                for dropout_rate in dropout_rate_list:
                    result = loop_exprienment(batch_size_train, n_epochs, learning_rate, dropout_rate)
                    print(batch_size_train, ", ",  n_epochs, ", ",  learning_rate, ", ", dropout_rate)
                    print(result)
                    csv_result_save(result)
    return


if __name__ == "__main__":
    main(sys.argv)

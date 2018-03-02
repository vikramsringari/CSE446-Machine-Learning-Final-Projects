import torch
import torchvision
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


trainset_dict = {}
testset_dict = {}
ordering = [line.strip() for line in open("competition-datasets-utf-8/projectgroup5_VANGOGHORNO-dataset/train-x")]

orderingtest = [line.strip() for line in open("competition-datasets-test-no-test-y/projectgroup5_VANGOGHORNO-dataset/test-x")]

for i in range(len(ordering)):
    trainset_dict[ordering[i]] = i

for i in range(len(orderingtest)):
    testset_dict[orderingtest[i]] = i



class ImageFolder2(datasets.ImageFolder):
    def __init__(self, root, transform):
       datasets.ImageFolder.__init__(self, root, transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path

'''os.makedirs("org_data")
os.makedirs("org_data/1")
os.makedirs("org_data/0")

ordering = [line.strip() for line in open("competition-datasets-utf-8/projectgroup5_VANGOGHORNO-dataset/train-x")]
labels = [line.strip() for line in open("competition-datasets-utf-8/projectgroup5_VANGOGHORNO-dataset/train-y")]

for i in range(len(ordering)):
    if(labels[i] == '0'):
        os.system("cp competition-datasets-utf-8/projectgroup5_VANGOGHORNO-dataset/train/" + str(ordering[i]) + " org_data/0/")
    if(labels[i] == '1'):
        os.system("cp competition-datasets-utf-8/projectgroup5_VANGOGHORNO-dataset/train/" + str(ordering[i]) + " org_data/1/")'''

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

art_dataset = ImageFolder2(root='org_data', transform=transform)
trainloader = torch.utils.data.DataLoader(art_dataset, batch_size=4, shuffle=True, num_workers=4)
test_art_dataset = ImageFolder2(root='org_test', transform=transform)
testloader = torch.utils.data.DataLoader(test_art_dataset, batch_size=1, shuffle=False, num_workers=4)

classes = ('0', '1')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

for epoch in range(25):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels, path = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

predlist = []

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
for data in testloader:
    images, labels, path = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(len(labels)):
        predlist.append((classes[predicted[i]], testset_dict[str(path[i].split("/")[-1])], str(path[i].split("/")[-1])))

sortedlist = sorted(predlist, key=lambda x: int(x[1]))
with open("40.5.test-yhat", "w") as f:
    for entry in sortedlist:
        f.write(entry[2] + "," + entry[0] + "\n")



'''def calc_accuracy(actual, prediction):
    a = [line.strip() for line in open(actual)]
    p = [line.strip() for line in open(prediction)]
    total = len(a)
    print(total)
    mistakes = 0
    for i in range(total):
        # print(str(a[i]) + " " + str(p[i]))
        if(a[i] != p[i]):
            mistakes += 1

    print("acc: " + str((total - mistakes)/total))
    


calc_accuracy("competition-datasets-utf-8/projectgroup5_VANGOGHORNO-dataset/train-y", "vango.test-yhat")'''




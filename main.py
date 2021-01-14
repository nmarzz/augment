# Import modules
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
import torchnet as tnt
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# Parser to help call
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--dropout', type=float, default=0.25, metavar='P',
                    help='dropout probability (default: 0.25)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='heavy ball momentum in gradient descent (default: 0.9)')
parser.add_argument('--data-dir', type=str, default='./data',metavar='DIR')
parser.add_argument('--dataset', type=str, default='FashionMNIST',metavar='DATA')
parser.add_argument('--track', type=bool, default=True,metavar='VIS',
                    help ='Track model training (Default: False)')
parser.add_argument('--augment', type=bool, default=False,metavar='AUG')
args = parser.parse_args()
args.cuda =  torch.cuda.is_available()



# Print out arguments to the log
print('Training LeNet on {}'.format(args.dataset))
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

if args.augment:
    train_transforms = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
else:
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])


###################
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
device = 'cuda' if args.cuda else 'cpu'
### Looks like we are using MNIST - (cause what else)
# Import train data

if args.dataset == 'MNIST':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=True,download = True,
                       transform=train_transforms),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    # Import test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=False, transform=test_transforms),
        batch_size=1000, shuffle=True, **kwargs)
elif args.dataset == 'FashionMNIST':
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(args.data_dir, train=True,download = True,
                       transform= train_transforms),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    # Import test data
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(args.data_dir, train=False, transform= test_transforms),
        batch_size=1000, shuffle=True, **kwargs)


##############
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

#  uuild our models
class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,args.dropout),
            convbn(20,50,5,2,args.dropout),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(args.dropout),
            nn.Linear(500,10))

    def forward(self, x):
        return self.m(x)

##############
model = LeNet()  # instantiate the model
loss_function = nn.CrossEntropyLoss() # use a CrossEntropyLoss function ie. right = 1,
if args.cuda:
    model.cuda()
    loss_function.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = args.momentum)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=5,threshold=1e-3,factor = 0.6)


if args.track:
    writer = SummaryWriter('runs/{}/{}'.format(args.dataset, 'augmented' if args.augment else 'standard'))
    test_images,y = next(iter(test_loader))

    # Save examples of the train images
    samples,labels = next(iter(train_loader))
    img_grid = torchvision.utils.make_grid(samples)
    matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image('{}_images'.format(args.dataset), img_grid)

    # toPil = transforms.Compose([transforms.ToPILImage(),transforms.Resize((128,128))])
    # toPil(vis_batch_data[0,:,:,:]).show()
    # y = visualization_batch[1].detach().cpu()
    # activations = {'final': [], 'y': y}






### Training loss_function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        if args.track and batch_idx % 100 == 0 and batch_idx>0:
            writer.add_scalar('training_loss',loss.data,epoch * len(train_loader) + batch_idx)

        if batch_idx % 100 == 0 and batch_idx>0:
            print('[Epoch %2d, batch %3d] training loss: %.4f' %
                (epoch, batch_idx, loss.data))

# A test function
def test():
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            loss = loss_function(output, target)

            top1.add(output.data, target.data)
            test_loss.add(loss.cpu().data)

    print('[Epoch %2d] Average test loss: %.3f, accuracy: %.2f%%\n'
        %(epoch, test_loss.value()[0], top1.value()[0]))

    if args.track:
        writer.add_scalar('test_loss',test_loss.value()[0],epoch)
        writer.add_scalar('test_acc', top1.value()[0],epoch)



# Actually run
weights = []
test_loss = tnt.meter.AverageValueMeter()
top1 = tnt.meter.ClassErrorMeter(accuracy = True)
if __name__=="__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # if args.track:
        #     model.eval()
        #     if args.cuda:
        #         vis_batch_data = vis_batch_data.cuda()
        #     output = model(vis_batch_data)
        #     activations['final'].append(output.detach().cpu())
        test()

    #
    # if args.track:
    #     path = 'activations{}_{}{}.pth'.format(args.dataset,args.epochs, 'augmented' if args.augment else '')
    #     activations['loss_meter'] = test_loss
    #     activations['top1'] = top1
    #     torch.save(activations,path)

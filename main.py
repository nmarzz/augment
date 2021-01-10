# Import modules
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchnet as tnt
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


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
parser.add_argument('--make-gif', type=bool, default=False,metavar='MKGIF')
args = parser.parse_args()
args.cuda =  torch.cuda.is_available()

# Print out arguments to the log
print('Training LeNet on MNIST')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')


###################
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
device = 'cuda' if args.cuda else 'cpu'
### Looks like we are using MNIST - (cause what else)
# Import train data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=True,download = True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
# Import test data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True, **kwargs)

##############
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

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
scheduler = optim.lr_scheduler.StepLR(optimizer,5)


if args.make_gif:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)

    tsne_images = []
    visualization_batch = next(iter(test_loader))
    vis_batch_data = visualization_batch[0]
    y = visualization_batch[1].detach()






### Training loss_function
def train(epoch):
    model.train()
    for batch_ix, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_ix % 100 == 0 and batch_ix>0:
            print('[Epoch %2d, batch %3d] training loss: %.4f' %
                (epoch, batch_ix, loss.data))

# A test function
def test():
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter(accuracy = True)
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

# Actually run
weights = []
if __name__=="__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if args.make_gif:
            model.eval()
            if args.cuda:
                vis_batch_data = vis_batch_data.cuda()
            output = model(vis_batch_data)
            res = tsne.fit_transform(output.detach())

            fig = plt.figure(figsize=(16,10))
            sns.scatterplot(x =res[:,0],y = res[:,1],palette=sns.color_palette("hls", 10),hue=y,legend='full')
            plt.title(str(epoch))
            img = fig2img(fig)
            tsne_images.append(img)
        test()

    if args.make_gif:
        img = tsne_images[0]
        img.save(fp='tsne_{}'.format(args.epochs), format='GIF', append_images=tsne_images[1:],
         save_all=True, duration=200, loop=0)

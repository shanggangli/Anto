
            import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import visdom


# parameter
viz=visdom.Visdom()
Batch_size=64
LR=0.001
EPOCH=10
hidden_size=30

train_dataset=datasets.MNIST('../../data/', True, transforms.ToTensor())
print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())
train_loader=DataLoader(train_dataset,Batch_size,True)

dataiter=iter(train_loader)
inputs, labels = dataiter.next()
#print(inputs.shape)
#viz.images(inputs[:4],opts=dict(title='input'))

# Anto_encoder
class Auto_encoder(nn.Module):
    def __init__(self):
        super(Auto_encoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,194),
            nn.Tanh(),
            nn.Linear(194,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3)
        )
        self.decoder=nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,194),
            nn.Tanh(),
            nn.Linear(194,28*28),
            nn.Sigmoid(),
        )
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded,decoded

antoencoder=Auto_encoder()
optimizer=torch.optim.Adam(antoencoder.parameters(),lr=LR)
loss_fuc=nn.MSELoss()

# train
for epoch in range(EPOCH):
    for step,(x,b_label) in enumerate(train_loader):
        b_x=x.view(-1,28*28)
        b_y=x.view(-1,28*28)

        encoded,decoded=antoencoder(b_x)
#loss and bp
        loss=loss_fuc(decoded,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#output and visdam
        if step % 100==0:
            print('Step',step,'| train loss: %.4f' % loss.data.numpy())
            # print(b_x[:2].view(-1,1,28,28))
            viz.images(b_x[:4].view(-1,1,28,28),opts=dict(title='input',width=300,height=100))
            viz.images(decoded[:4].view(-1,1,28,28),opts=dict(title='decoded',width=300,height=100))

    

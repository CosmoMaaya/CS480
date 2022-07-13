from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

if not os.path.exists('gan_results'):
    os.mkdir('gan_results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
criterion = nn.BCELoss()
real_label = 1.
fake_label = 0.

class Generator(nn.Module):
    #The generator takes an input of size latent_size, and will produce an output of size 784.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.layer(z)

class Discriminator(nn.Module):
    #The discriminator takes an input of size 784, and will produce an output of size 1.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    #Trains both the generator and discriminator for one epoch on the training dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    #TODO
    generator.train()
    discriminator.train()
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        discriminator_optimizer.zero_grad()
        x_real= x.view(-1, 784).to(device)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output = discriminator(x_real).view(-1)
        lossD_real = criterion(output, label)
        lossD_real.backward()


        z = torch.randn(batch_size, latent_size, device=device)
        x_fake = generator(z)
        label.fill_(fake_label)
        output = discriminator(x_fake.detach()).view(-1)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()
        label.fill_(real_label)
        output = discriminator(x_fake).view(-1)
        lossG = criterion(output, label)
        lossG.backward()
        generator_optimizer.step()

        avg_generator_loss += lossG.item()
        avg_discriminator_loss += lossD.item()
 
    avg_generator_loss = avg_generator_loss / len(train_loader)
    avg_discriminator_loss = avg_discriminator_loss / len(train_loader)

    return avg_generator_loss, avg_discriminator_loss

def test(generator, discriminator):
    #Runs both the generator and discriminator over the test dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    #TODO
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x_real= x.view(-1, 784).to(device)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(x_real).view(-1)
            lossD_real = criterion(output, label)


            z = torch.randn(batch_size, latent_size, device=device)
            x_fake = generator(z)
            label.fill_(fake_label)
            output = discriminator(x_fake.detach()).view(-1)
            lossD_fake = criterion(output, label)

            lossD = lossD_real + lossD_fake

            label.fill_(real_label)
            output = discriminator(x_fake).view(-1)
            lossG = criterion(output, label)

            avg_generator_loss += lossG.item()
            avg_discriminator_loss += lossD.item()
        
        avg_generator_loss = avg_generator_loss / len(train_loader)
        avg_discriminator_loss = avg_discriminator_loss / len(train_loader)

    return avg_generator_loss, avg_discriminator_loss

if __name__ == '__main__':
    epochs = 50

    discriminator_avg_train_losses = []
    discriminator_avg_test_losses = []
    generator_avg_train_losses = []
    generator_avg_test_losses = []

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
        generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

        discriminator_avg_train_losses.append(discriminator_avg_train_loss)
        generator_avg_train_losses.append(generator_avg_train_loss)
        discriminator_avg_test_losses.append(discriminator_avg_test_loss)
        generator_avg_test_losses.append(generator_avg_test_loss)

        with torch.no_grad():
            sample = torch.randn(64, latent_size).to(device)
            sample = generator(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                    'gan_results/sample_' + str(epoch) + '.png')
            print('Epoch #' + str(epoch))
            display(Image('gan_results/sample_' + str(epoch) + '.png'))
            print('\n')

    plt.plot(discriminator_avg_train_losses)
    plt.plot(generator_avg_train_losses)
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Disc','Gen'], loc='upper right')
    plt.savefig('gan_results/train_loss.png')
    plt.show()

    plt.plot(discriminator_avg_test_losses)
    plt.plot(generator_avg_test_losses)
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Disc','Gen'], loc='upper right')
    plt.savefig('gan_results/test_loss.png')
    plt.show()

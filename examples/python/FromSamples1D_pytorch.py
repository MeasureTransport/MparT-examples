# # Pytorch Integration
# This example demonstrates how to train and use an MParT-based transport map 
# in a larger pytorch model.  To illustrate, a regression problem with heteroscedastic
# noise will be considered.  
#
# The goal of this problem is to characterize a conditional distribution $p(y|x)$,
# where $x,y\in\mathbb{R}$.  To characterize this distribution, we will construct 
# a conditional transport map of the form $T(y; f(x))$, where $f:\mathbb{R}\rightarrow\mathbb{R}^M$
# is a feature extractor that returns an $M$ dimensional vector.  This feature extractor 
# will be defined with a neural network (pytorch) while the map itself $T:\mathbb{R}^{M+1}\rightarrow\mathbb{R}$
# is defined with a polynomial-based monotone map (MParT).

import matplotlib.pyplot as plt
import torch
import mpart as mt


# ## Generate Training Data 

num_pts = 1000
x_train = torch.linspace(-2.0*torch.pi, 2.0*torch.pi, num_pts, dtype=torch.double)
y_true = torch.sin(x_train)
y_train = y_true + 0.5*torch.abs(1.0+torch.cos(x_train))*torch.randn(num_pts)

dataset = torch.utils.data.TensorDataset(x_train.reshape(-1,1), y_train.reshape(-1,1)) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
num_epochs = 200


# ##  Define the hybrid model

maxDegree = 3 # <- The maximum total order of the map $T$
tgt_dim = 1 # <- The dimension of the target random variable $y$
cond_dim = 2 # <- The number of features $M$ returned by the neural network

# #### MParT map Construction
# Here we construct the 3->1 dimensional map in the typical fashion of MParT but 
# using options that help make the map robust to inputs.

opts = mt.MapOptions()
opts.basisLB = -2.5
opts.basisUB = 2.5
opts.nugget = 0.001
tmap = mt.CreateTriangular(cond_dim+tgt_dim, tgt_dim, maxDegree, opts) # Simple third order map

# #### Hybrid Construction
# We can now create a torch model for the composition $T(y; f(x))$.  The `.torch()` function
# from the MParT map returns a torch.nn.Module that can be used as a native pytorch object.
# The autograd functionality in pytorch can also be used with this function.

class MapModel(torch.nn.Module):
    def __init__(self, tmap):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 300, dtype=torch.double)
        self.activation = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(300, cond_dim, dtype=torch.double)
        self.tmap = tmap.torch(return_logdet=True)

    def forward(self, x, y):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        x = torch.hstack([x,y])
        r, logdet = self.tmap.forward(x)

        return r, logdet

    def inverse(self, x, r):
        """ This is an additional function for computing the inverse of $T$ to compute $y$ from $r$.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = torch.hstack([x,r])

        y = self.tmap.inverse(x,r)

        return y


model = MapModel(tmap)

# ## Train the model
# Here we use the negative log likelihood as a training function.
#
# Given a value of $x$, the hybrid function $r=T(y; f(x))$ defines a monotone transformation
# that can be used to transform probability distributions.  By assuming the distribution 
# of $r$ is $\mu_r$, we can define a map-induced distribution over $y$ given by the pull back
# of the reference measure $\mu_r$ through the map: $\mu_y=T^{\sharp}\mu_r$.  The log density of this 
# distribution is proportinoal to $\log p( T(y; f(x)) ) + \log \text{det}\nabla_y T$, where $
# $p(r)$ is the density of the reference distribution.  To compute the parameters of the map and 
# the weights of the neural network, we will maximize the likelihood of the training data according
# to this map-induced density and the assumption that $p(r)=N(0,1)$ is standard normal.


model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=7e-3)
num_epochs = 200
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        
        x_batch, y_batch = data

        ref, logdet = model.forward(x_batch, y_batch)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Evaluate the map and its log determinant for this batch
        ref, logdet = model(x_batch, y_batch)

        # Compute the negative log likelihood as a loss function
        loss = -torch.mean(-0.5*ref*ref + logdet)  # <- proportional to negative log likelihood with standard normal reference distribution
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        if (i % 200) == 0:
            print(f'Loss: {loss.item():0.4e}')

# Now that we've finished training, we can go ahead with plotting

# ## Plot the results
model.eval()

# We plot the true function along with the training data accordingly.

plt.plot(x_train,y_train,'.', label='Training Data',alpha=0.6, zorder=2)
plt.plot(x_train,y_true,'-k',label='True $f(x)$',linewidth=3, zorder=2)
plt.legend()

num_plot = 200 
x_plot = torch.linspace(-2.0*torch.pi, 2.0*torch.pi, num_plot, dtype=torch.double).reshape(-1,1)

# Since the map is monotone and real-valued, we can find the median and the quantiles directly

# Compute the median.
r = torch.zeros((num_plot,1), dtype=torch.double)
median = model.inverse(x_plot,r)

# Compute the 5% quantile 
r = -2.0*torch.ones((num_plot,1), dtype=torch.double)
q05 = model.inverse(x_plot,r)

# Compute the 95% quantile 
r = 2.0*torch.ones((num_plot,1), dtype=torch.double)
q95 = model.inverse(x_plot,r)


plt.fill_between(x_plot.ravel(),q05.ravel(),q95.ravel(), color='b', alpha=0.1, label='%5-95% CI', zorder=1)

# Finally, let's plot the median as well as some realizations of the noise

num_samples_per_point = 10
r = torch.randn((num_plot*num_samples_per_point,1), dtype=torch.double)
xplot_rep = x_plot.repeat(num_samples_per_point, 1)
samples = model.inverse(xplot_rep,r)

plt.plot(x_plot, median, 'b', label='Median', linewidth=2, zorder=3)
plt.scatter(xplot_rep, samples, s=1, color='b', alpha=0.1, label='Samples', zorder=1)
plt.legend()
plt.show()

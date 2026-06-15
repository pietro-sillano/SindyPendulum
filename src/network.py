import torch
import torch.nn as nn

from sindy_library import SINDyLibrary

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, input_size,latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,latent_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))   
        x = torch.relu(self.fc4(x))
        return x 

class Decoder(nn.Module):
    def __init__(self, input_size,latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,input_size)
        self.initialize_weights()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))   
        x = torch.relu(self.fc4(x))
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size,latent_dim)
        self.decoder = Decoder(input_size,latent_dim)
        self.SINDyLibrary = SINDyLibrary(
            device=device,
            latent_dim=latent_dim,
            include_biases=True,
            include_states=True,
            include_sin=True,
            include_cos=True,
            include_multiply_pairs=True,
            poly_order=2,
            include_sqrt=False,
            include_inverse=False,
            include_sign_sqrt_of_diff=False)
        

        self.XI = nn.Parameter(torch.full((self.SINDyLibrary.number_candidate_functions,latent_dim),1,dtype = torch.float32,requires_grad=True,device = device))

        self.XI_coefficient_mask = torch.ones((self.SINDyLibrary.number_candidate_functions,latent_dim),dtype = torch.float32, device=device)

        self.mse = nn.MSELoss()

        enc_params = list(self.encoder.parameters())
        self._enc_weights = [w for w in enc_params if len(w.shape) == 2]
        self._enc_biases  = [b for b in enc_params if len(b.shape) == 1]
        dec_params = list(self.decoder.parameters())
        self._dec_weights = [w for w in dec_params if len(w.shape) == 2]
        self._dec_biases  = [b for b in dec_params if len(b.shape) == 1]

    def configure_optimizers(self):
        learning_rate = 1e-4
 #       return torch.optim.SGD(self.parameters(), lr=0.1)
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


    def t_derivative(self,input, xdot, weights, biases, activation='sigmoid'):
        """
        Compute the first order time derivatives by propagating through the network.
        da[l]dt = xdot * da[l]dx = xdot * product(g'(w[l]a[l-1] + b[l])* w[l])
        Arguments:
            input - 2D tensorflow array, input to the network. Dimensions are number of time points
            by number of state variables.
            xdot - First order time derivatives of the input to the network. quello che conosciamo
            weights - List of tensorflow arrays containing the network weights
            biases - List of tensorflow arrays containing the network biases
            activation - String specifying which activation function to use. Options are
            'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
            or linear.

        Returns:
            dadt - Tensorflow array, first order time derivatives of the network output.
        """
        a   = input
        dadt = xdot #per le condizioni iniziali

        if activation == 'sigmoid':
            for i in range(len(weights) - 1):
                z = torch.matmul(a, weights[i].T) + biases[i]
                a = torch.sigmoid(z)
                gprime = a * (1-a)
                dadt = gprime * torch.matmul(dadt, weights[i].T)
            dadt = torch.matmul(dadt, weights[-1].T) #fuori dal ciclo bisogna ancora moltiplicare per i pesi dell ultimo livello
            
        elif activation == 'relu':
            for i in range(len(weights) - 1):
                z = torch.matmul(a, weights[i].T) + biases[i]
                a = torch.relu(z)
                dadt = (z > 0).float() * torch.matmul(dadt, weights[i].T)    
            dadt = torch.matmul(dadt, weights[-1].T) #fuori dal ciclo bisogna ancora moltiplicare per i pesi dell ultimo livello
        return dadt #nel caso che ci serve dadt sará l output dell encoder ossia le latent variables!

    
    
    def compute_quantities(self,x,xdot):
    
        z = self.encoder(x)
        xtilde = self.decoder(z)

        theta = self.SINDyLibrary.transform(z)
        zdot_hat = torch.matmul(theta, self.XI_coefficient_mask * self.XI)

        zdot = self.t_derivative(x, xdot, self._enc_weights, self._enc_biases, activation='relu')
        xtildedot = self.t_derivative(z, zdot_hat, self._dec_weights, self._dec_biases, activation='relu')
        
        return xtilde, xtildedot, z, zdot, zdot_hat

    def loss_function(self, x, xdot, xtilde, xtildedot, zdot, zdot_hat, XI):
        alpha1 = 5e-4
        alpha2 = 5e-5
        alpha3 = 1e-5
        loss = {}
        loss['recon_loss'] = self.mse(x, xtilde)
        loss['sindy_loss_x'] = self.mse(xdot, xtildedot)
        loss['sindy_loss_z'] = self.mse(zdot, zdot_hat)
        loss['sindy_regular_loss'] = torch.sum(torch.abs(XI))
        loss['tot'] = (loss['recon_loss']
                       + alpha1 * loss['sindy_loss_x']
                       + alpha2 * loss['sindy_loss_z']
                       + alpha3 * loss['sindy_regular_loss'])
        return loss['tot'], loss
    
    def forward(self, x, xdot):
        return self.compute_quantities(x, xdot)
    
    
    



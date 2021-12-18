class Encoder(nn.Module):
    def __init__(self, input_size,latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,latent_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))    
        return x 

class Decoder(nn.Module):
    def __init__(self, input_size,latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim,64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,input_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))    
        return x 

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size,latent_dim)
        self.decoder = Decoder(input_size,latent_dim)
        self.sindy_library()
        self.XI = nn.Parameter(torch.full((self.SINDyLibrary.number_candidate_functions, latent_dim), 0.1))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

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
        dadt = xdot #per le condizioni iniziali
        if activation == 'sigmoid':

            for i in range(len(model.stack)-1):
                weights = model.stack[i].weight
                biases = model.stack[i].bias
                
                z = torch.matmul(input, weights[i]) + biases[i]

                a = torch.sigmoid(z)
                
                gprime = torch.mul(a, 1-a)

                dadt = torch.mul(gprime,torch.matmul(dadt, weights[i]))

            dadt = torch.matmul(dadt, weights[-1]) #fuori dal ciclo bisogna ancora moltiplicare per i pesi dell ultimo livello

        return dadt #nel caso che ci serve dadt sará l output dell encoder ossia le latent variables!

    def compute_quantities(self,x,xdot):
        z = self.encoder(x)

        xtilde = self.decoder(z)

        theta = self.SINDyLibrary.transform(z) # rimane da definire questa parte
        if self.sequential_thresholding:
            zdot_hat = torch.matmul(theta, self.XI_coefficient_mask * self.XI)
        else:
            zdot_hat = torch.matmul(theta, self.XI)


        encoder_parameters = list(self.encoder.parameters())
        encoder_weight_list = [w for w in encoder_parameters if len(w.shape) == 2]
        encoder_biases_list = [b for b in encoder_parameters if len(b.shape) == 1]
        
        zdot = self.t_derivative(self,input, xdot, encoder_weight_list, encoder_biases_list, activation='sigmoid')                                               



        decoder_parameters = list(self.decoder.parameters())
        decoder_weight_list = [w for w in decoder_parameters if len(w.shape) == 2]
        decoder_biases_list = [b for b in decoder_parameters if len(b.shape) == 1]

        #zdot_hat é z ricostruito grazie a sindy
        xtildedot = self.t_derivative(self,z, zdot_hat, decoder_weight_list, decoder_biases_list, activation='sigmoid')                                               
        return xtilde, xtildedot, z, zdot, zdot_hat

    def loss_function(self, x, xdot, xtilde, xtildedot, zdot, zdot_hat, XI):
        mse = nn.MSELoss()
        alpha = 0.01 #regolarizer parameters
        recon_loss = mse(x, x_hat) #errore di ricostruzione 

        sindy_loss_x = mse(xdot, xtildedot) 

        sindy_loss_z = mse(zdot, zdot_hat) #dove zdot hat é la ricostruzione di z con sindy

        sindy_regular_loss = torch.sum(torch.abs(XI)) #norma L1 degli XI

        loss = recon_loss + sindy_loss_x + sindy_loss_z \
               + alpha * sindy_regular_loss
        return loss
    
    
    def forward(self, x, xdot):
        return self.compute_quantities(x, xdot)
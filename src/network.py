import torch
import torch.nn as nn

from sindy_library import SINDyLibrary

device = "cuda" if torch.cuda.is_available() else "cpu"

_PENDULUM_SINDY_CONFIG = dict(
    include_biases=False,
    include_states=True,
    include_sin=True,
    include_cos=False,
    include_multiply_pairs=False,
    poly_order=1,
    include_sqrt=False,
    include_inverse=False,
    include_sign_sqrt_of_diff=False,
)


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, input_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim, sindy_config=None):
        super().__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(input_size, latent_dim)

        cfg = sindy_config if sindy_config is not None else _PENDULUM_SINDY_CONFIG
        self.SINDyLibrary = SINDyLibrary(device=device, latent_dim=latent_dim, **cfg)

        n_funcs = self.SINDyLibrary.number_candidate_functions
        self.XI = nn.Parameter(
            torch.ones(n_funcs, latent_dim, dtype=torch.float32, device=device)
        )
        self.XI_coefficient_mask = torch.ones(
            n_funcs, latent_dim, dtype=torch.float32, device=device
        )
        self.mse = nn.MSELoss()

        enc_params = list(self.encoder.parameters())
        self._enc_weights = [w for w in enc_params if w.dim() == 2]
        self._enc_biases  = [b for b in enc_params if b.dim() == 1]
        dec_params = list(self.decoder.parameters())
        self._dec_weights = [w for w in dec_params if w.dim() == 2]
        self._dec_biases  = [b for b in dec_params if b.dim() == 1]

    def t_derivative(self, input, xdot, weights, biases, activation='relu'):
        """
        Jacobian-vector product: propagates xdot = dx/dt through the network
        to produce da/dt at the output layer.

        Uses the chain rule layer by layer:
            da_l/dt = g'(z_l) * (W_l @ da_{l-1}/dt)
        where z_l = W_l @ a_{l-1} + b_l.
        """
        a    = input
        dadt = xdot

        if activation == 'sigmoid':
            for i in range(len(weights)):
                z = torch.matmul(a, weights[i].T) + biases[i]
                a = torch.sigmoid(z)
                dadt = a * (1 - a) * torch.matmul(dadt, weights[i].T)

        elif activation == 'relu':
            for i in range(len(weights)):
                z = torch.matmul(a, weights[i].T) + biases[i]
                a = torch.relu(z)
                dadt = (z > 0).float() * torch.matmul(dadt, weights[i].T)

        return dadt

    def compute_quantities(self, x, xdot):
        z      = self.encoder(x)
        xtilde = self.decoder(z)

        theta    = self.SINDyLibrary.transform(z)
        zdot_hat = torch.matmul(theta, self.XI_coefficient_mask * self.XI)

        zdot      = self.t_derivative(x, xdot, self._enc_weights, self._enc_biases)
        xtildedot = self.t_derivative(z, zdot_hat, self._dec_weights, self._dec_biases)

        return xtilde, xtildedot, z, zdot, zdot_hat

    def loss_function(self, x, xdot, xtilde, xtildedot, zdot, zdot_hat, XI):
        alpha1 = 5e-4
        alpha2 = 5e-5
        alpha3 = 1e-5
        loss = {}
        loss['recon_loss']         = self.mse(x, xtilde)
        loss['sindy_loss_x']       = self.mse(xdot, xtildedot)
        loss['sindy_loss_z']       = self.mse(zdot, zdot_hat)
        loss['sindy_regular_loss'] = torch.sum(torch.abs(XI))
        loss['tot'] = (loss['recon_loss']
                       + alpha1 * loss['sindy_loss_x']
                       + alpha2 * loss['sindy_loss_z']
                       + alpha3 * loss['sindy_regular_loss'])
        return loss['tot'], loss

    def forward(self, x, xdot):
        return self.compute_quantities(x, xdot)

import torch
from torch import nn

class MDNDecisionMaker(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout_prob):
        super().__init__()
        self.hidden = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout_prob
        self.feed_forward_network = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, self.hidden), #1
            nn.SiLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(self.hidden, self.hidden), #2
            nn.SiLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(self.hidden, self.hidden), #3
            nn.SiLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(self.hidden, self.hidden), #3
            nn.SiLU(),
            nn.BatchNorm1d(self.hidden)
        ).double()
        # predict mean value of multivariate gaussian distribution
        self.mean_network = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), #3
            nn.SiLU(),
            nn.Linear(self.hidden, out_dim)
        ).double()
        # predict non diagonal lower triangular values of matrix
        self.cholesky_nondiag_sigmas_network = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), #3
            nn.SiLU(),
            nn.Linear(self.hidden, out_dim*out_dim), #2
        ).double()
        # predict the diagonal elements, these must be non zero to ensure invertibility
        self.cholesky_diag_sigmas_network = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), #3
            nn.SiLU(),
            nn.Linear(self.hidden, out_dim)
        ).double()
        self.bceloss = nn.BCELoss()

    def forward(self, x, return_covariance = False):
        parameters = self.feed_forward_network(x.double())
        means = self.mean_network(parameters)
        cholesky_lower_triangular = torch.tril(self.cholesky_nondiag_sigmas_network(parameters).view(-1, self.out_dim, self.out_dim), diagonal = -1)
        cholesky_diag = torch.diag_embed(torch.exp(self.cholesky_diag_sigmas_network(parameters)).view(-1, self.out_dim))
        cholesky_sigmas =  cholesky_diag + cholesky_lower_triangular
        if return_covariance:
            covariances = torch.bmm(cholesky_sigmas, torch.transpose(cholesky_sigmas, 1, 2))
            return mean, covariances
        return means, cholesky_sigmas
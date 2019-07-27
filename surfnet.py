from __future__ import print_function
import torch
import numpy as np
from unet import UnaryNet, PairNet

# define the clapping threshold for probability normalization.
STAB_NB = 1e-15


def newton_sol_pd(g_mean, g_sigma, w_comp, d_p):
    '''
    This function solves the quadratic CRF: 1/2 xT H x + pT x. Assume g_mean has shape: bn,  x_len. w_comp is a torch parameter.
    The size of d_p is one less than g_mean, since we currently do not consider the head and tail difference.
    '''
    x_len = g_mean.size(1)
    # The Hessian is divided into two parts: pairwise and unary.
    hess_pair = torch.diag(-2.*w_comp.repeat(x_len-1), diagonal=-1) + torch.diag(-2.*w_comp.repeat(x_len-1),
                    diagonal=1) + torch.diag(torch.cat((2.*w_comp, 4.*w_comp.repeat(x_len-2), 2.*w_comp), dim=0),
                        diagonal=0)
    # hess_pair = torch.stack(hess_pair)
    # pairwise parts are the same across patches within a batch
    if g_mean.is_cuda:
        hess_pair_batch = torch.stack([hess_pair]*g_sigma.size(0)).cuda()
    else:
        hess_pair_batch = torch.stack([hess_pair]*g_sigma.size(0))
    # get reverse of sigma array
    g_sigma_rev = 1./g_sigma
    # convert sigma reverse array to diagonal matrices
    g_sigma_eye = torch.diag_embed(g_sigma_rev)
    # sum up two parts
    hess_batch = hess_pair_batch + g_sigma_eye
    # compute inverse of Hessian
    hess_inv_batch = torch.inverse(hess_batch)
    # generate the linear coefficient P
    p_u = g_mean/g_sigma
    
    # print(p_u.size(), d_p.size())
    delta = 2.*(torch.cat((d_p, torch.zeros(d_p.size(0), 1).cuda()), dim=-1) - 
                                        torch.cat((torch.zeros(d_p.size(0), 1).cuda(), d_p), dim=-1))
    p = p_u + delta
    # solve it globally
    solution = torch.matmul(hess_inv_batch, p.unsqueeze(-1)).squeeze(-1)

    return solution

def normalize_prob(x):
    '''Normalize prob map to [0, 1]. Numerically, add 1e-6 to all. Assume the last dimension is prob map.'''
    x_norm = (x - x.min(-1, keepdim=True)
              [0]) / (x.max(-1, keepdim=True)[0] - x.min(-1, keepdim=True)[0])
    x_norm += 1e-3
    return x_norm


def gaus_fit(x, tr_flag=True):
    '''This module is designed to regress Gaussian function. Weighted version is chosen. The input tensor should
    have the format: BN,  X_LEN, COL_LEN.'''
    bn,  x_len, col_len = tuple(x.size())
    col_ind_set = torch.arange(col_len).expand(
        bn,  x_len, col_len).double()
    if x.is_cuda:
        col_ind_set = col_ind_set.cuda()
    y = x.double()
    lny = torch.log(y).double()
    y2 = torch.pow(y, 2).double()
    x2 = torch.pow(col_ind_set, 2).double()
    sum_y2 = torch.sum(y2, dim=-1)
    sum_xy2 = torch.sum(col_ind_set * y2, dim=-1)
    sum_x2y2 = torch.sum(x2 * y2, dim=-1)
    sum_x3y2 = torch.sum(torch.pow(col_ind_set, 3) * y2, dim=-1)
    sum_x4y2 = torch.sum(torch.pow(col_ind_set, 4) * y2, dim=-1)
    sum_y2lny = torch.sum(y2 * lny, dim=-1)
    sum_xy2lny = torch.sum(col_ind_set * y2 * lny, dim=-1)
    sum_x2y2lny = torch.sum(x2 * y2 * lny, dim=-1)
    b_num = (sum_x2y2**2*sum_xy2lny - sum_y2*sum_x4y2*sum_xy2lny + sum_xy2*sum_x4y2*sum_y2lny +
             sum_y2*sum_x3y2*sum_x2y2lny - sum_x2y2*sum_x3y2*sum_y2lny - sum_xy2*sum_x2y2*sum_x2y2lny)
    c_num = (sum_x2y2lny*sum_xy2**2 - sum_xy2lny*sum_xy2*sum_x2y2 - sum_x3y2*sum_y2lny*sum_xy2 +
             sum_y2lny*sum_x2y2**2 - sum_y2*sum_x2y2lny*sum_x2y2 + sum_y2*sum_x3y2*sum_xy2lny)
    c_num[(c_num < STAB_NB) & (c_num > -STAB_NB)
          ] = torch.sign(c_num[(c_num < STAB_NB) & (c_num > -STAB_NB)]) * STAB_NB
    mu = -b_num / (2.*c_num)

    c_din = sum_x4y2*sum_xy2**2 - 2*sum_xy2*sum_x2y2*sum_x3y2 + \
        sum_x2y2**3 - sum_y2*sum_x4y2*sum_x2y2 + sum_y2*sum_x3y2**2
    sigma_b_sqrt = -0.5*c_din/c_num
    sigma_b_sqrt[sigma_b_sqrt < 1] = 1
    sigma = sigma_b_sqrt
    #TODO May have better strategies to handle the failure of Gaussian fitting.
    if not tr_flag:
        mu[mu >= col_len-1] = col_len-1
        mu[mu <= 0] = 0.
    if torch.isnan(mu).any() or torch.isnan(sigma).any():
        raise Exception("mu or sigma gets NaN value.")

    mu = mu.float()
    sigma = sigma.float()

    return mu, sigma

class SurfSegNet(torch.nn.Module):
    """
    ONly GPU version has been implemented!!!
    """
    def __init__(self, unary_model, wt_init=1e-5,  pair_model=None):
        super(SurfSegNet, self).__init__()
        self.unary = unary_model
        self.pair = pair_model
        self.w_comp = torch.nn.Parameter(torch.ones(1)*wt_init)
    def forward(self, x, tr_flag=False):
        logits = self.unary(x).squeeze(1).permute(0, 2, 1)  
        logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = normalize_prob(logits)
        if self.pair is None:
            d_p = torch.zeros((x.size(0), x.size(-1)-1), dtype=torch.float32, requires_grad=False).cuda()
        else:
            self.pair.eval()
            d_p = self.pair(x)
     
        mean, sigma = gaus_fit(logits, tr_flag=tr_flag)
        output = newton_sol_pd(mean, sigma, self.w_comp, d_p)

        return output

if __name__ == "__main__":
    unary_model = UnaryNet(num_classes=1, in_channels=1, depth=5, start_filts=1, up_mode="bilinear")
    pair_model = PairNet(num_classes=1, in_channels=1, depth=5, start_filts=1, up_mode="bilinear")
    surfnet = SurfSegNet(unary_model=unary_model).cuda()
    x = torch.FloatTensor(np.random.random((2,1,512,400))).cuda()
    y = surfnet(x)
    print(y.size())

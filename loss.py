import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
        self,
        prediction='v',
        path_type="linear",
        weighting="uniform",
        encoders=[], 
        accelerator=None, 
        latents_scale=None, 
        latents_bias=None,
    ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError("The specified path type is not implemented.")

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_proj_loss(self, zs, zs_tilde):
        proj_loss = 0.
        bsz = zs[0].shape[0]
        if zs is not None and len(zs) > 0:
            for (z, z_t) in zip(zs, zs_tilde):
                    for (z_j, z_t_j) in zip(z, z_t):
                        z_j = F.normalize(z_j, dim=-1)
                    z_t_j = F.normalize(z_t_j, dim=-1)
                    proj_loss += mean_flat(-(z_j * z_t_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)
        else:
            proj_loss = torch.tensor(0., device=images.device, dtype=images.dtype)
        return proj_loss       

    def forward_warmup(self, model, images, model_kwargs, zs):
        _, zs_tilde = model(images, torch.zeros_like(images[:, 0, 0, 0]).flatten(), **model_kwargs)
        proj_loss = self.compute_proj_loss(zs, zs_tilde)
        return None, proj_loss    

    def forward_full(self, model, images, model_kwargs, zs):
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)        

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        return denoising_loss, proj_loss

    def __call__(self, model, images, model_kwargs=None, zs=None, warmup=False):
        if model_kwargs == None:
            model_kwargs = {}
        if warmup:
            return self.forward_warmup(model, images, model_kwargs, zs)
        else:
            return self.forward_full(model, images, model_kwargs, zs)

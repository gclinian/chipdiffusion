import pos_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tfd
import numpy as np
import networks
from omegaconf import open_dict
import orientations
import guidance
import os
import schedulers

# TODO apply pos-encoding refactor for Res-MLP and ViT
class DiffusionModel(nn.Module):
    backbones = {"mlp": networks.ConditionalMLP, "res_mlp": networks.ResidualMLP, "unet": networks.UNet, "cond_unet": networks.CondUNet, "vit": networks.ViT}
    time_encodings = {"sinusoid": pos_encoding.get_positional_encodings, "none": pos_encoding.get_none_encodings}

    def __init__(self, backbone, backbone_params, in_channels, image_size, encoding_type, encoding_dim, max_diffusion_steps = 100, noise_schedule = "linear", device = "cpu", **kwargs):
        super().__init__()
        if backbone == "mlp" or backbone == "res_mlp":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_size": in_channels * image_size[0] * image_size[1],
                    "out_size": in_channels * image_size[0] * image_size[1],
                    "encoding_dim": encoding_dim,
                    "device": device,
                })
        elif backbone == "unet" or backbone == "cond_unet":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": in_channels,
                    "out_channels": in_channels,
                    "image_shape": (image_size[0], image_size[1]),
                    "cond_dim": encoding_dim,
                    "device": device,
                })
        elif backbone == "vit":
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": in_channels,
                    "out_channels": in_channels,
                    "image_size": image_size[0],
                    "encoding_dim": encoding_dim,
                    "device": device,
                })
        if encoding_dim > 0:
            self.encoding = DiffusionModel.time_encodings[encoding_type](max_diffusion_steps, encoding_dim).to(device)
        self.encoding_dim = encoding_dim
        self._reverse_model = DiffusionModel.backbones[backbone](**backbone_params)
        self.in_channels = in_channels
        self.max_diffusion_steps = max_diffusion_steps
        self.image_size = image_size
        if noise_schedule == "linear":
            beta = get_linear_sched(max_diffusion_steps, kwargs["beta_1"], kwargs["beta_T"])
            self._alpha_bar = torch.tensor(compute_alpha(beta), device = device, dtype=torch.float32)
            self._beta = torch.tensor(beta, device = device, dtype=torch.float32)
        else:
            raise NotImplementedError
        
        self._loss = nn.MSELoss(reduction = "mean")

        # cache some variables:
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_alpha_bar_complement = torch.sqrt(1 - self._alpha_bar)
        self._epsilon_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self._sigma = torch.sqrt(self._beta)

    def __call__(self, x, t):
        # input: x is (B, C, H, W) for images, t is (B)
        # output: epsilon predictions of model
        t_embed = self.compute_pos_encodings(t)
        return self._reverse_model(x, t_embed).view(*x.shape)
    
    def loss(self, x, t):
        B = x.shape[0] # x is (B, C, H, W) for images
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        # sample epsilon and generate noisy images
        epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1) # (B, C, H, W)
        x = self._sqrt_alpha_bar[t-1].view(B, *([1]*len(x.shape[1:]))) * x + self._sqrt_alpha_bar_complement[t-1].view(B, *([1]*len(x.shape[1:]))) * epsilon
        x = self(x, t)
        metrics = {"epsilon_theta_mean": x.detach().mean().cpu().numpy(), "epsilon_theta_std": x.detach().std().cpu().numpy()}
        return self._loss(x, epsilon), metrics
    
    def forward_samples(self, x, intermediate_every = 0):
        intermediates = [x]
        for t in range(self.max_diffusion_steps):
            epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1)
            x_t = self._sqrt_alpha_bar[t] * x + self._sqrt_alpha_bar_complement[t] * epsilon
            if intermediate_every and t<(self.max_diffusion_steps-1) and t % intermediate_every == 0:
                intermediates.append(x_t)
        intermediates.append(x_t) # append final image
        return intermediates

    def reverse_samples(self, B, intermediate_every = 0):
        # B: batch size
        # intermediate_every: determines how often intermediate diffusion steps are saved and returned. 0 = no intermediates returned
        batch_shape = (B, self.in_channels, self.image_size[0], self.image_size[1])
        x = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) # (B, C, H, W)
        intermediates = [x]
        for t in range(self.max_diffusion_steps, 0 , -1):
            t_vec = torch.tensor(t, device=x.device).expand(B)
            z = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if t>1 else torch.zeros_like(x)
            x = (1.0/torch.sqrt(1 - self._beta[t-1])) * (x - (self._beta[t-1]/self._sqrt_alpha_bar_complement[t-1]) * self(x, t_vec))
            if intermediate_every and t>1 and t % intermediate_every == 0:
                intermediates.append(x)
            x = x + self._sigma[t-1] * z
        intermediates.append(x) # append final image
        # normalize x
        x_max = torch.amax(x, dim=(1,2,3), keepdim=True)
        x_min = torch.amin(x, dim=(1,2,3), keepdim=True)
        x_normalized = (x - x_min)/(x_max - x_min)
        return x_normalized, intermediates
    
    def compute_pos_encodings(self, t):
        # t has shape (B,)
        if self.encoding_dim == 0:
            return None
        B = t.shape[0]
        encoding = self.encoding[t-1, :].view(B, self.encoding_dim)
        return encoding

class CondDiffusionModel(nn.Module):
    backbones = {
        "mlp": networks.ConditionalMLP, 
        "res_mlp": networks.ResidualMLP, 
        "unet": networks.UNet, 
        "cond_unet": networks.CondUNet, 
        "vit": networks.ViT, 
        "res_gnn": networks.ResGNN, 
        "res_gnn_block": networks.ResGNNBlock, 
        "graph_unet": networks.GraphUNet,
        "att_gnn": networks.AttGNN, # Current best
        "graph_transformer": networks.GraphTransformer,
        }
    time_encodings = {
        "sinusoid": pos_encoding.get_positional_encodings, 
        "none": pos_encoding.get_none_encodings,
        }
    # conditioning vec can be arbitrary
    # here we use a torch_geometry object
    def __init__(
            self, 
            backbone, 
            backbone_params, 
            input_shape, 
            t_encoding_type, 
            t_encoding_dim, 
            max_diffusion_steps = 100, 
            noise_schedule = "linear", 
            mask_key = None, 
            use_mask_as_input = False, 
            device = "cpu", 
            **kwargs
            ):
        super().__init__()
        if backbone == "mlp" or backbone == "res_mlp":
            self.modality = "image"
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_size": np.prod(input_shape),
                    "out_size": np.prod(input_shape),
                    "encoding_dim": t_encoding_dim,
                    "device": device,
                })
        elif backbone == "unet" or backbone == "cond_unet":
            self.modality = "image"
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": input_shape[0],
                    "out_channels": input_shape[0],
                    "image_shape": (input_shape[1], input_shape[2]),
                    "cond_dim": t_encoding_dim,
                    "device": device,
                })
        elif backbone == "vit":
            self.modality = "image"
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": input_shape[0],
                    "out_channels": input_shape[0],
                    "image_size": input_shape[1],
                    "encoding_dim": t_encoding_dim,
                    "device": device,
                })
        elif backbone in ["res_gnn_block", "res_gnn", "graph_unet", "att_gnn", "graph_transformer"]:
            self.modality = "graph"
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_node_features": input_shape[1],
                    "out_node_features": input_shape[1],
                    "t_encoding_dim": t_encoding_dim,
                    "device": device,
                    "mask_key": mask_key if use_mask_as_input else None,
                })
        if t_encoding_dim > 0:
            self.t_encoding = CondDiffusionModel.time_encodings[t_encoding_type](max_diffusion_steps, t_encoding_dim).to(device) # (N_steps, C)
        self.mask_key = mask_key
        self.t_encoding_dim = t_encoding_dim
        self._reverse_model = CondDiffusionModel.backbones[backbone](**backbone_params)
        self.input_shape = input_shape
        self.max_diffusion_steps = max_diffusion_steps
        if noise_schedule == "linear":
            beta = get_linear_sched(max_diffusion_steps, kwargs["beta_1"], kwargs["beta_T"])
            self._alpha_bar = torch.tensor(compute_alpha(beta), device = device, dtype=torch.float32) # (N_steps,)
            self._beta = torch.tensor(beta, device = device, dtype=torch.float32) # (N_steps,)
        else:
            raise NotImplementedError
        
        self._lossfn = nn.MSELoss(reduction = "mean")

        # cache some variables:
        self.device = device
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar) # (N_steps,)
        self._sqrt_alpha_bar_complement = torch.sqrt(1 - self._alpha_bar)
        self._epsilon_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self._sigma = torch.sqrt(self._beta)

    def __call__(self, x, cond, t):
        # input: x is (B, C, H, W) for images, t is (B), cond is (1, x)
        # note: 1 graph at a time
        # output: epsilon predictions of model
        t_embed = self.compute_pos_encodings(t)
        return self._reverse_model(x, cond, t_embed).view(*x.shape)
    
    def loss(self, x, cond, t):
        B = x.shape[0] # x is (B, C, H, W) for images, (B, V, F) for graphs
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)
        # sample epsilon and generate noisy images
        epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1) # (B, C, H, W) or (B, V, F)
        x_perturbed = self._sqrt_alpha_bar[t-1].view(B, *([1]*len(x.shape[1:]))) * x + self._sqrt_alpha_bar_complement[t-1].view(B, *([1]*len(x.shape[1:]))) * epsilon
        # don't perturb things covered by mask
        x = torch.where(mask, x, x_perturbed) if mask is not None else x_perturbed
        x = self(x, cond, t)
        x_masked = x.detach()[torch.logical_not(mask).expand(x.shape)] if mask is not None else x.detach()
        metrics = {"epsilon_theta_mean": x_masked.mean().cpu().numpy(), "epsilon_theta_std": x_masked.std().cpu().numpy()}
        return self._loss(x, epsilon, mask), metrics
    
    def forward_samples(self, x, cond, intermediate_every = 0):
        intermediates = [x]
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)
        for t in range(self.max_diffusion_steps):
            epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1)
            x_t = self._sqrt_alpha_bar[t] * x + self._sqrt_alpha_bar_complement[t] * epsilon
            if mask is not None: # don't perturb things covered by mask
                x_t = torch.where(mask, x, x_t)
            if intermediate_every and t<(self.max_diffusion_steps-1) and t % intermediate_every == 0:
                intermediates.append(x_t)
        intermediates.append(x_t) # append final image
        return intermediates

    def reverse_samples(self, B, x_in, cond, intermediate_every = 0, mask_override = None, output_log_prob = False):
        # B: batch size
        # intermediate_every: determines how often intermediate diffusion steps are saved and returned. 0 = no intermediates returned
        if self.modality == "image":
            batch_shape = (B, *self.input_shape)
            mask_shape = (1, x_in.shape[1], 1, 1)
        else: # graphs
            batch_shape = (B, cond.x.shape[0], self.input_shape[1])
            mask_shape = (1, x_in.shape[1], 1)
        x = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) # (B, C, H, W)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x
        intermediates = [x]
        if output_log_prob:
            log_probs = torch.zeros((self.max_diffusion_steps, B,), device = x.device)

        for t in range(self.max_diffusion_steps, 0 , -1):
            t_vec = torch.tensor(t, device=x.device).expand(B)
            z = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if t>1 else torch.zeros_like(x)
            mu = (1.0/torch.sqrt(1 - self._beta[t-1])) * (x - (self._beta[t-1]/self._sqrt_alpha_bar_complement[t-1]) * self(x, cond, t_vec))
            # don't denoise things covered by mask
            mu = torch.where(mask, x, mu) if mask is not None else mu
            if intermediate_every and t>1 and t % intermediate_every == 0:
                intermediates.append(mu.detach())
            x_perturbed = mu.detach() + self._sigma[t-1] * z
            # don't perturb things covered by mask
            x = torch.where(mask, x, x_perturbed) if mask is not None else x_perturbed
            if output_log_prob:
                log_probs[t-1, :] = pi_log_prob(x, mu, self._sigma[t-1])

        intermediates.append(x) # append final image
        # normalize x
        if self.modality == "image":
            x_max = torch.amax(x, dim=(1,2,3), keepdim=True)
            x_min = torch.amin(x, dim=(1,2,3), keepdim=True)
            x_normalized = (x - x_min)/(x_max - x_min)
        else:
            x_normalized = x
        
        if output_log_prob:
            return x_normalized, intermediates, log_probs
        else:
            return x_normalized, intermediates
    
    def compute_pos_encodings(self, t):
        # t has shape (B,)
        if self.t_encoding_dim == 0:
            return None
        B = t.shape[0]
        encoding = self.t_encoding[t-1, :].view(B, self.t_encoding_dim)
        return encoding

    def get_mask(self, x, cond):
        if self.modality == "graph":
            if self.mask_key and self.mask_key in cond: # TODO raise error if mask key unexpectedly missing
                mask = cond[self.mask_key]
                B, V, F = x.shape
                mask = mask.view(1, V, 1)
                return mask
            else:
                return None
        else:
            raise NotImplementedError
        
    def _loss(self, x, target, mask = None):
        if mask is not None:
            numel = torch.numel(mask)
            squared_error = torch.square(x-target)
            squared_error.masked_fill_(mask, 0)
            mse = torch.mean(squared_error) * (numel / (numel - torch.sum(mask)))
            return mse
        else:
            return self._lossfn(x, target)

class SkipDiffusionModel(CondDiffusionModel):
    def __init__(
            self,
            max_diffusion_steps = 100,
            base_diffusion_steps = 1000, 
            **kwargs
            ):
        super().__init__(**kwargs, max_diffusion_steps = max_diffusion_steps)
        assert base_diffusion_steps % max_diffusion_steps == 0, "new diffusion steps must divide base steps"
        self.base_diffusion_steps = base_diffusion_steps
        self.speedup_factor = base_diffusion_steps // max_diffusion_steps
        dummy_model = CondDiffusionModel(**kwargs, max_diffusion_steps = base_diffusion_steps)
        
        self.t_encoding = self.remap_tensor(dummy_model.t_encoding)
        self._alpha_bar = self.remap_tensor(dummy_model._alpha_bar)
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_alpha_bar_complement = torch.sqrt(1 - self._alpha_bar)
        
        alpha = torch.zeros_like(self._alpha_bar)
        alpha[1:] = self._alpha_bar[1:] / self._alpha_bar[:-1]
        alpha[0] = self._alpha_bar[0]

        self._beta = 1-alpha
        self._sigma = torch.sqrt(self._beta)

    def remap_tensor(self, x):
        # Given x tensor(T_base, ...)
        # Return tensor(T_new, ...) where timesteps are sampled from
        in_shape = x.shape
        x_reshape = x.view(self.max_diffusion_steps, self.speedup_factor, *in_shape[1:])
        return x_reshape[:, -1] # Note: this is actually quite inelegant because of the endpoints, esp large speedup factors
    
class GuidedDiffusionModel(CondDiffusionModel):
    # Uses potential-based guidance for sampling
    def __init__(
            self, 
            backbone, 
            backbone_params, 
            input_shape, 
            t_encoding_type, 
            t_encoding_dim,
            legality_guidance_weight,
            hpwl_guidance_weight,
            guidance_step,
            forward_guidance_weight,
            grad_descent_steps,
            grad_descent_rate,
            self_recursion_steps, 
            max_diffusion_steps = 100, 
            noise_schedule = "linear", 
            mask_key = None, 
            use_mask_as_input = False,
            device = "cpu", 
            **kwargs
        ):
        self.legality_guidance_weight = legality_guidance_weight
        self.hpwl_guidance_weight = hpwl_guidance_weight
        self.guidance_step = guidance_step
        self.forward_guidance_weight = forward_guidance_weight
        self.grad_descent_steps = grad_descent_steps # denoted m in universal guidance paper
        self.grad_descent_rate = grad_descent_rate
        self.self_recursion_steps = self_recursion_steps # denoted k in universal guidance paper
        super().__init__(
            backbone, 
            backbone_params, 
            input_shape, 
            t_encoding_type, 
            t_encoding_dim, 
            max_diffusion_steps = max_diffusion_steps, 
            noise_schedule = noise_schedule, 
            mask_key = mask_key, 
            use_mask_as_input = use_mask_as_input, 
            device = device, 
            **kwargs
        )
    
    def reverse_samples(self, B, x_in, cond, intermediate_every = 0, mask_override = None, output_log_prob = False):
        # B: batch size
        # intermediate_every: determines how often intermediate diffusion steps are saved and returned. 0 = no intermediates returned
        DEBUG = False # TODO change this!!

        if self.modality == "image":
            batch_shape = (B, *self.input_shape)
            mask_shape = (1, x_in.shape[1], 1, 1)
        else: # graphs
            batch_shape = (B, cond.x.shape[0], self.input_shape[1])
            mask_shape = (1, x_in.shape[1], 1)
        x = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) # (B, C, H, W)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x
        intermediates = [x]
        if output_log_prob:
            log_probs = torch.zeros((self.max_diffusion_steps, B,), device = x.device)
        
        if DEBUG:
            import utils
            grad_norms = []
            grad_norms_std = []

        for t in range(self.max_diffusion_steps, 0 , -1):
            x_prev = x
            z = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if t>1 else torch.zeros_like(x)
            g, eps_predict = self.guidance_force_xhat(x, cond, t, mask=mask)
            if output_log_prob: # need gradients on eps
                t_vec = torch.tensor(t, device=x.device).expand(B)
                eps_predict = self(x, cond, t_vec)
            mu = (1.0/torch.sqrt(1 - self._beta[t-1])) * (x - (self._beta[t-1]/self._sqrt_alpha_bar_complement[t-1]) * eps_predict)
            
            # don't denoise things covered by mask
            mu = torch.where(mask, x_prev, mu) if mask is not None else mu
            if intermediate_every and (t-1) % intermediate_every == 0: # TODO get rid of this
                x_hat = (x - self._sqrt_alpha_bar_complement[t-1] * eps_predict.detach())/self._sqrt_alpha_bar[t-1]
                intermediates.append(x_hat)
                # intermediates.append(mu.detach())

            if DEBUG and t % 200 <= 1:
                # debugging
                x_hat = (x - self._sqrt_alpha_bar_complement[t-1] * eps_predict.detach())/self._sqrt_alpha_bar[t-1] #(B, V, 2)
                image = utils.visualize_placement(x_hat[0], cond, plot_pins=False, plot_edges=False, img_size=(256, 256))
                file_idx = cond.file_idx if "file_idx" in cond else 0
                utils.debug_plot_img(image, os.path.join("debug", f"xhat_{file_idx}_{t}"))
                image = utils.visualize_placement(mu[0], cond, plot_pins=False, plot_edges=False, img_size=(256, 256))
                utils.debug_plot_img(image, os.path.join("debug", f"beforeguidance_{file_idx}_{t}"))
                
            if t <= self.guidance_step:
                mu = mu + g
                # g_norm = torch.mean(torch.norm(g, dim=2, keepdim=True))
                # mu = mu + self._sqrt_alpha_bar[t-1] * (g/max(g_norm, 1)) # stronger guidance for smaller t
            # if intermediate_every and (t-1) % intermediate_every == 0:
            #     intermediates.append(mu.detach())

            if DEBUG:
                g_norm = torch.norm(g, dim=2)
                grad_norms.append(torch.mean(g_norm).item())
                grad_norms_std.append(torch.std(g_norm).item())
                if t % 200 <= 1:
                    # debugging
                    image = utils.visualize_placement(mu[0], cond, plot_pins=False, plot_edges=False, img_size=(256, 256))
                    utils.debug_plot_img(image, os.path.join("debug", f"afterguidance_{file_idx}_{t}"))

            x = mu.detach() + self._sigma[t-1] * z
            x = torch.clamp(x, -2, 2) # avoid nans TODO revert to -4, 4
            # don't perturb things covered by mask
            x = torch.where(mask, x_prev, x) if mask is not None else x
            if output_log_prob:
                log_probs[t-1, :] = pi_log_prob(x, mu, self._sigma[t-1])
        
        if DEBUG:
            utils.debug_plot_graph(grad_norms, os.path.join("debug", f"guidance_norms"), fig_title="guidance norm")
            utils.debug_plot_graph(grad_norms_std, os.path.join("debug", f"guidance_norm_std"), fig_title="guidance std")
            import ipdb; ipdb.set_trace()

        # normalize x
        if self.modality == "image":
            x_max = torch.amax(x, dim=(1,2,3), keepdim=True)
            x_min = torch.amin(x, dim=(1,2,3), keepdim=True)
            x_normalized = (x - x_min)/(x_max - x_min)
        else:
            x_normalized = x
        
        # outputs
        if output_log_prob:
            # TODO debug remove
            print(x_in.shape)
            import utils; utils.debug_mem()
            return x_normalized, intermediates, log_probs
        else:
            return x_normalized, intermediates

    def guidance_force(self, x, cond, mask=None):
        # gives conditional guidance force g (see Dhariwal & Nichol 2021) NOTE this is deprecated
        # mask (1, V, 1) 
        # x (B, V, 2)
        B, V, D = x.shape
        sizes = cond.x.expand(B, *cond.x.shape) # (B, V, D)
        x = x + sizes/2 # centered

        x_1 = x.view(B, V, 1, D)
        x_2 = x.view(B, 1, V, D)
        size_1 = sizes.view(B, V, 1, D)
        size_2 = sizes.view(B, 1, V, D)

        # compute pairwise signed distance function
        delta = torch.abs(x_1 - x_2) - ((size_1 + size_2)/2) # (B, V1, V2, D)
        l, max_idx = torch.max(delta, dim=-1, keepdim=True)
        dl_dx = F.one_hot(max_idx.squeeze(dim=-1), num_classes=D) * torch.sign(x_1-x_2)
        g = F.relu(-l) * dl_dx
        
        mask_square = (1-torch.eye(V, dtype=g.dtype, device=g.device)).view(1, V, V, 1) # ignore self-collision
        if mask is not None:
            inv_mask = ~mask
            mask_square = mask_square * inv_mask.view(1, 1, V, 1) * inv_mask.view(1, V, 1, 1)
        g = mask_square * g
        
        # scale by geometric mean of instance dimensions
        mass_1 = torch.exp(torch.mean(torch.log(sizes), dim=-1, keepdim=True)).unsqueeze(dim=-1) # (B, V1, 1, 1)
        mass_2 = mass_1.view(B, 1, V, 1) # (B, 1, V2, 1)
        g = g * (mass_2/(mass_1 + mass_2)) # (B, V1, V2, D)
        g_sum = g.sum(dim=-2)  # (B, V1, D)

        # apply boundary forces from chip canvas edges
        g_bound = -F.relu(torch.abs(x) + sizes/2 - 1) * torch.sign(x) # (B, V, D)
        g_bound = inv_mask * g_bound if mask is not None else g_bound

        g_sum = g_sum + g_bound
        return g_sum
    
    @torch.enable_grad()
    def guidance_force_xhat(self, x, cond, t, mask=None, eps_grad=False):
        # gives conditional guidance force g (see Dhariwal & Nichol 2021)
        # this gives the guidance term for universal backwards guidance
        # mask (1, V, 1)
        # x (B, V, 2)
        # TODO Cache pin maps for hpwl guidance
        # Note eps_grad uses more mem than just computing forward pass twice
        B, V, D = x.shape
        # TODO make these tunable options
        legality_softmax_factor_min = 10.0
        legality_softmax_factor_max = 10.0
        legality_softmax_factor = legality_softmax_factor_max - (t/self.max_diffusion_steps) * (legality_softmax_factor_max-legality_softmax_factor_min)
        
        x.requires_grad_(True)
        t_vec = torch.tensor(t, device=x.device).expand(B)
        try:
            self.requires_grad_(eps_grad)
            eps_predict = self(x, cond, t_vec)
            self.requires_grad_(True)
        except:
            self.requires_grad_(True)
            raise
        x_hat = (x - self._sqrt_alpha_bar_complement[t-1] * eps_predict)/self._sqrt_alpha_bar[t-1]
        
        # compute backward guidance on x_hat
        x_hat_current = x_hat.detach().clone().requires_grad_(True)
        optimizer = torch.optim.SGD((x_hat_current,), lr=self.grad_descent_rate, momentum=0.0)
        # m step gradient descent TODO stop movements of objects that should be masked out
        for _ in range(self.grad_descent_steps):
            optimizer.zero_grad()
            h_legality = self.legality_guidance_weight * guidance.legality_guidance_potential(x_hat_current, cond, mask=mask, softmax_factor=legality_softmax_factor)
            h_hpwl = self.hpwl_guidance_weight * guidance.hpwl_guidance_potential(x_hat_current, cond)
            h = h_legality + h_hpwl
            h.sum().backward()
            optimizer.step()
        
        delta_x_hat = x_hat_current.detach() - x_hat.detach()
        g = delta_x_hat * self._sqrt_alpha_bar[t-1]
        eps_predict = eps_predict if eps_grad else eps_predict.detach()

        if self.forward_guidance_weight > 0:
            h_legality = self.legality_guidance_weight * guidance.legality_guidance_potential(x_hat, cond, mask=mask, softmax_factor=legality_softmax_factor)
            h_hpwl = self.hpwl_guidance_weight * guidance.hpwl_guidance_potential(x_hat, cond)
            h = h_legality + h_hpwl
            h.sum().backward(retain_graph=eps_grad)
            g = g.detach() - self.forward_guidance_weight * x.grad * self._sqrt_alpha_bar[t-1]
            x.requires_grad_(False)
        
        return g.detach(), eps_predict

class SkipGuidedDiffusionModel(GuidedDiffusionModel):
    def __init__(
            self,
            max_diffusion_steps = 100,
            base_diffusion_steps = 1000, 
            **kwargs
            ):
        super().__init__(**kwargs, max_diffusion_steps = max_diffusion_steps)
        assert base_diffusion_steps % max_diffusion_steps == 0, "new diffusion steps must divide base steps"
        self.base_diffusion_steps = base_diffusion_steps
        self.speedup_factor = base_diffusion_steps // max_diffusion_steps
        dummy_model = GuidedDiffusionModel(**kwargs, max_diffusion_steps = base_diffusion_steps)
        
        self.t_encoding = self.remap_tensor(dummy_model.t_encoding)
        self._alpha_bar = self.remap_tensor(dummy_model._alpha_bar)
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_alpha_bar_complement = torch.sqrt(1 - self._alpha_bar)
        
        alpha = torch.zeros_like(self._alpha_bar)
        alpha[1:] = self._alpha_bar[1:] / self._alpha_bar[:-1]
        alpha[0] = self._alpha_bar[0]

        self._beta = 1-alpha
        self._sigma = torch.sqrt(self._beta)

    def remap_tensor(self, x):
        # Given x tensor(T_base, ...)
        # Return tensor(T_new, ...) where timesteps are sampled from
        in_shape = x.shape
        x_reshape = x.view(self.max_diffusion_steps, self.speedup_factor, *in_shape[1:])
        return x_reshape[:, -1] # Note: this is actually quite inelegant because of the endpoints, esp large speedup factors

class SelfCondDiffusionModel(nn.Module):
    backbones = {
        "mlp": networks.ConditionalMLP, 
        "res_mlp": networks.ResidualMLP, 
        "unet": networks.UNet, 
        "cond_unet": networks.CondUNet, 
        "vit": networks.ViT, 
        "res_gnn": networks.ResGNN, 
        "res_gnn_block": networks.ResGNNBlock, 
        "graph_unet": networks.GraphUNet,
        "att_gnn": networks.AttGNN, # Current best
        "graph_transformer": networks.GraphTransformer,
        }
    time_encodings = {
        "sinusoid": pos_encoding.get_positional_encodings, 
        "none": pos_encoding.get_none_encodings,
        }
    # conditioning vec can be arbitrary
    # here we use a torch_geometry object
    def __init__(
            self, 
            backbone, 
            backbone_params, 
            input_shape, 
            t_encoding_type, 
            t_encoding_dim, 
            self_cond_mode = "model_output",
            max_diffusion_steps = 100, 
            noise_schedule = "linear", 
            mask_key = None, 
            use_mask_as_input = False, 
            device = "cpu", 
            **kwargs
            ):
        super().__init__()
        if backbone == "mlp" or backbone == "res_mlp":
            self.modality = "image"
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_size": 2 * np.prod(input_shape),
                    "out_size": np.prod(input_shape),
                    "encoding_dim": t_encoding_dim,
                    "device": device,
                })
        elif backbone == "unet" or backbone == "cond_unet":
            self.modality = "image"
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": 2 * input_shape[0],
                    "out_channels": input_shape[0],
                    "image_shape": (input_shape[1], input_shape[2]),
                    "cond_dim": t_encoding_dim,
                    "device": device,
                })
        elif backbone == "vit":
            self.modality = "image"
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_channels": 2 * input_shape[0],
                    "out_channels": input_shape[0],
                    "image_size": input_shape[1],
                    "encoding_dim": t_encoding_dim,
                    "device": device,
                })
        elif backbone in ["res_gnn_block", "res_gnn", "graph_unet", "att_gnn", "graph_transformer"]:
            self.modality = "graph"
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_node_features": 2 * input_shape[1],
                    "out_node_features": input_shape[1],
                    "t_encoding_dim": t_encoding_dim,
                    "device": device,
                    "mask_key": mask_key if use_mask_as_input else None,
                })
        if t_encoding_dim > 0:
            self.t_encoding = CondDiffusionModel.time_encodings[t_encoding_type](max_diffusion_steps, t_encoding_dim).to(device)
        self.mask_key = mask_key
        self.t_encoding_dim = t_encoding_dim
        self._reverse_model = CondDiffusionModel.backbones[backbone](**backbone_params)
        self.input_shape = input_shape
        self.max_diffusion_steps = max_diffusion_steps
        if noise_schedule == "linear":
            beta = get_linear_sched(max_diffusion_steps, kwargs["beta_1"], kwargs["beta_T"])
            self._alpha_bar = torch.tensor(compute_alpha(beta), device = device, dtype=torch.float32)
            self._beta = torch.tensor(beta, device = device, dtype=torch.float32)
        else:
            raise NotImplementedError
        
        self._lossfn = nn.MSELoss(reduction = "mean")

        assert self_cond_mode in ["model_output", "x0"], "self conditioning mode must be set to either 'model_output' or 'x0'"
        self.self_cond_mode = self_cond_mode

        # cache some variables:
        self.device = device
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_alpha_bar_complement = torch.sqrt(1 - self._alpha_bar)
        self._epsilon_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self._sigma = torch.sqrt(self._beta)

    def __call__(self, x, cond, t):
        # input: x is (B, C, H, W) for images, t is (B), cond is (1, x)
        # note: 1 graph at a time
        # output: epsilon predictions of model
        t_embed = self.compute_pos_encodings(t)
        output = self._reverse_model(x, cond, t_embed)
        return output
    
    def loss(self, x, cond, t):
        metrics = {}
        B = x.shape[0] # x is (B, C, H, W) for images, (B, V, F) for graphs
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)
        
        # sample epsilon and generate noisy images
        epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1) # (B, C, H, W) or (B, V, F)
        x_perturbed = self._sqrt_alpha_bar[t-1].view(B, *([1]*len(x.shape[1:]))) * x + self._sqrt_alpha_bar_complement[t-1].view(B, *([1]*len(x.shape[1:]))) * epsilon
        # don't perturb things covered by mask
        x_t = torch.where(mask, x, x_perturbed) if mask is not None else x_perturbed
        
        # self-conditioning
        null_condition = torch.zeros_like(x_t)
        x_input = torch.concat((x_t, null_condition), dim = -1)
        model_output = self(x_input, cond, t)
        
        is_self_condition = torch.randint(2, size=(1,)).item()
        if is_self_condition:
            model_output.detach_()
            null_cond_loss = self._loss(model_output, epsilon, mask)

            if self.self_cond_mode == "model_output":
                x_input = torch.concat((x_t, model_output), dim = -1)
            elif self.self_cond_mode == "x0":
                x_0 = (x_t.detach() - self._sqrt_alpha_bar_complement[t-1].view(B, *([1]*len(x.shape[1:]))) * model_output)/torch.sqrt(1 - self._sqrt_alpha_bar_complement[t-1]**2).view(B, *([1]*len(x.shape[1:])))
                x_0 = torch.clamp(x_0, -3, 3) # avoid nans
                x_0 = torch.where(mask, x, x_0) if mask is not None else x_0 # mask out ports
                x_input = torch.concat((x_t, x_0), dim = -1)

            model_output = self(x_input, cond, t)
            
            train_loss = self._loss(model_output, epsilon, mask)
            metrics["self_cond_loss"] = train_loss.detach().cpu().numpy()
        else:
            null_cond_loss = self._loss(model_output, epsilon, mask)
            train_loss = null_cond_loss
        
        output_masked = model_output.detach()[torch.logical_not(mask).expand(model_output.shape)] if mask is not None else model_output.detach()
        metrics.update({
            "epsilon_theta_mean": output_masked.mean().cpu().numpy(), 
            "epsilon_theta_std": output_masked.std().cpu().numpy(),
            "null_cond_loss": null_cond_loss.detach().cpu().numpy(),
            })
        
        return train_loss, metrics
    
    def forward_samples(self, x, cond, intermediate_every = 0):
        intermediates = [x]
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)
        for t in range(self.max_diffusion_steps):
            epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1)
            x_t = self._sqrt_alpha_bar[t] * x + self._sqrt_alpha_bar_complement[t] * epsilon
            if mask is not None: # don't perturb things covered by mask
                x_t = torch.where(mask, x, x_t)
            if intermediate_every and t<(self.max_diffusion_steps-1) and t % intermediate_every == 0:
                intermediates.append(x_t)
        intermediates.append(x_t) # append final image
        return intermediates

    def reverse_samples(self, B, x_in, cond, intermediate_every = 0, mask_override = None):
        # B: batch size
        # intermediate_every: determines how often intermediate diffusion steps are saved and returned. 0 = no intermediates returned
        if self.modality == "image":
            batch_shape = (B, *self.input_shape)
            mask_shape = (1, x_in.shape[1], 1, 1)
        else: # graphs
            batch_shape = (B, cond.x.shape[0], self.input_shape[1])
            mask_shape = (1, x_in.shape[1], 1)
        x = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) # (B, C, H, W)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x
        self_condition = torch.zeros_like(x)
        intermediates = [x]
        
        for t in range(self.max_diffusion_steps, 0 , -1):
            t_vec = torch.tensor(t, device=x.device).expand(B)
            z = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if t>1 else torch.zeros_like(x)
            
            x_input = torch.concat((x, self_condition), dim = -1)
            eps_predict = self(x_input, cond, t_vec)
            if self.self_cond_mode == "model_output":
                self_condition = eps_predict
            elif self.self_cond_mode == "x0":
                self_condition = (x.detach() - self._sqrt_alpha_bar_complement[t-1] * eps_predict.detach())/torch.sqrt(1 - self._sqrt_alpha_bar_complement[t-1]**2)
                self_condition = torch.clamp(self_condition, -3, 3) # avoid nans
                self_condition = torch.where(mask, x_in, self_condition) if mask is not None else self_condition # mask out ports
            else:
                raise NotImplementedError
            
            x_denoised = (1.0/torch.sqrt(1 - self._beta[t-1])) * (x - (self._beta[t-1]/self._sqrt_alpha_bar_complement[t-1]) * eps_predict)
            # don't denoise things covered by mask
            x = torch.where(mask, x, x_denoised) if mask is not None else x_denoised
            if intermediate_every and t>1 and t % intermediate_every == 0:
                intermediates.append(x)
            x_perturbed = x + self._sigma[t-1] * z
            # don't perturb things covered by mask
            x = torch.where(mask, x, x_perturbed) if mask is not None else x_perturbed
        
        intermediates.append(x) # append final image
        # normalize x
        if self.modality == "image":
            x_max = torch.amax(x, dim=(1,2,3), keepdim=True)
            x_min = torch.amin(x, dim=(1,2,3), keepdim=True)
            x_normalized = (x - x_min)/(x_max - x_min)
        else:
            x_normalized = x
        return x_normalized, intermediates
    
    def compute_pos_encodings(self, t):
        # t has shape (B,)
        if self.t_encoding_dim == 0:
            return None
        B = t.shape[0]
        encoding = self.t_encoding[t-1, :].view(B, self.t_encoding_dim)
        return encoding

    def get_mask(self, x, cond):
        if self.modality == "graph":
            if self.mask_key and self.mask_key in cond: 
                mask = cond[self.mask_key]
                B, V, F = x.shape
                mask = mask.view(1, V, 1)
                return mask
            else:
                return None
        else:
            raise NotImplementedError
        
    def _loss(self, x, target, mask = None):
        if mask is not None:
            numel = torch.numel(mask)
            squared_error = torch.square(x-target)
            squared_error.masked_fill_(mask, 0)
            mse = torch.mean(squared_error) * (numel / (numel - torch.sum(mask)))
            return mse
        else:
            return self._lossfn(x, target)

class MixedDiffusionModel(nn.Module):
    """
    Diffusion model for mixing continuous and discrete spaces
    Specific to graph modality
    Continuous input has shape (V, F_cont), discrete input has shape (V, F_disc)
    """
    backbones = {
        "att_gnn": networks.AttGNN, # Current best
        "graph_transformer": networks.GraphTransformer,
        }
    time_encodings = {
        "sinusoid": pos_encoding.get_positional_encodings, 
        "none": pos_encoding.get_none_encodings,
        }
    def __init__(
            self, 
            backbone, 
            backbone_params, 
            input_cont_dim,
            input_disc_dim,
            t_encoding_type, 
            t_encoding_dim, 
            self_cond_mode = "model_output",
            discrete_prediction = "x0",
            max_diffusion_steps = 100, 
            noise_schedule = "linear", 
            mask_key = None, 
            use_mask_as_input = False, 
            device = "cpu", 
            **kwargs
            ):
        super().__init__()
        if backbone in ["att_gnn", "graph_transformer"]:
            with open_dict(backbone_params):
                backbone_params.update({
                    "in_node_features": 2 * (input_cont_dim + input_disc_dim), # for self-conditioning
                    "out_node_features": input_cont_dim + input_disc_dim,
                    "t_encoding_dim": t_encoding_dim,
                    "device": device,
                    "mask_key": mask_key if use_mask_as_input else None,
                })
        if t_encoding_dim > 0:
            self.t_encoding = CondDiffusionModel.time_encodings[t_encoding_type](max_diffusion_steps, t_encoding_dim).to(device)
        self.mask_key = mask_key
        self.t_encoding_dim = t_encoding_dim
        self._reverse_model = CondDiffusionModel.backbones[backbone](**backbone_params)
        self.input_cont_dim = input_cont_dim
        self.input_disc_dim = input_disc_dim

        self.max_diffusion_steps = max_diffusion_steps
        if noise_schedule == "linear":
            beta = get_linear_sched(max_diffusion_steps, kwargs["beta_1"], kwargs["beta_T"])
            self._alpha_bar = torch.tensor(compute_alpha(beta), device = device, dtype=torch.float32)
            self._beta = torch.tensor(beta, device = device, dtype=torch.float32)
        else:
            raise NotImplementedError
        
        self._lossfn = nn.MSELoss(reduction = "mean")

        if self_cond_mode is not None:
            assert self_cond_mode in ["model_output", "x0", "none"], "self conditioning mode must be set to either 'model_output' or 'x0' or none"
        self.self_cond_mode = self_cond_mode
        assert discrete_prediction in ["x0", "epsilon"], "discrete_prediction must be either 'x0' or 'epsilon'"
        self.discrete_prediction = discrete_prediction

        # cache some variables:
        self.device = device
        self._sqrt_alpha_bar = torch.sqrt(self._alpha_bar)
        self._sqrt_alpha_bar_complement = torch.sqrt(1 - self._alpha_bar)
        self._epsilon_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self._sigma = torch.sqrt(self._beta)

    def __call__(self, x_cont, x_disc, cond, t):
        """
        input: x is (B, V, F) for graphs (for continuous and disc inputs), t is (B), cond is pyG data object.
        note: 1 graph at a time
        output: epsilon predictions of model
        """
        t_embed = self.compute_t_encodings(t)
        x_combined = self.combine_inputs(x_cont, x_disc)
        output = self._reverse_model(x_combined, cond, t_embed)
        output_cont, output_disc = self.split_inputs(output)
        return output_cont, output_disc
    
    def loss(self, x_cont, x_disc, cond, t):
        """
        x_cont: (B, V, F_cont)
        x_disc: (B, V, F_disc)
        cond: PyG Data object containing x (B, V), edge_index, edge_attr, mask
        t: (B)
        """
        metrics = {}
        B, V, _ = x_cont.shape # x is (B, V, F) for graphs
        assert t.shape[0] == B and len(t.shape) == 1, "t has to have shape (B)"
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(cond)
        coeff_shape = [B] + [1] * len(x_cont.shape[1:]) # for reshaping alpha_bar, etc.

        # sample epsilon and generate noisy images
        epsilon_cont = self._epsilon_dist.sample(x_cont.shape).squeeze(dim = -1) # (B, V, F)
        epsilon_disc = self._epsilon_dist.sample(x_disc.shape).squeeze(dim = -1) # (B, V, F)
        x_perturbed_cont = self._sqrt_alpha_bar[t-1].view(*coeff_shape) * x_cont + self._sqrt_alpha_bar_complement[t-1].view(*coeff_shape) * epsilon_cont
        x_perturbed_disc = self._sqrt_alpha_bar[t-1].view(*coeff_shape) * x_disc + self._sqrt_alpha_bar_complement[t-1].view(*coeff_shape) * epsilon_disc
        # don't perturb things covered by mask
        x_t_cont = torch.where(mask, x_cont, x_perturbed_cont) if mask is not None else x_perturbed_cont
        x_t_disc = torch.where(mask, x_disc, x_perturbed_disc) if mask is not None else x_perturbed_disc
        
        # self-conditioning TODO check for edge cases in terms of t?
        null_condition_cont = torch.zeros_like(x_t_cont)
        null_condition_disc = torch.zeros_like(x_t_disc)
        x_input_cont = torch.concat((x_t_cont, null_condition_cont), dim = -1)
        x_input_disc = torch.concat((x_t_disc, null_condition_disc), dim = -1)
        model_output_cont, model_output_disc = self(x_input_cont, x_input_disc, cond, t)

        # prediction target for continuous is epsilon, for discrete is x0
        target_cont = epsilon_cont
        target_disc = x_disc if self.discrete_prediction=="x0" else epsilon_disc

        # debugging TODO remove or refactor
        bitwise_loss = self._loss(model_output_disc, target_disc, mask, debug=True).detach().cpu().numpy() # (3)
        if self.discrete_prediction == "x0":
            pred_x0 = model_output_disc
        else:
            pred_x0 = (x_t_disc - self._sqrt_alpha_bar_complement[t-1].view(*coeff_shape) * model_output_disc.detach())/torch.sqrt(1 - self._sqrt_alpha_bar_complement[t-1]**2).view(*coeff_shape)
        bitwise_accuracies = torch.clamp(torch.sgn(pred_x0 * x_disc), min=0).mean(dim=1).mean(dim=0).detach().cpu().numpy() # check sign equality
        metrics.update({
            "bit_0_loss": bitwise_loss[0],
            "bit_1_loss": bitwise_loss[1],
            "bit_2_loss": bitwise_loss[2],
            "bit_0_acc": bitwise_accuracies[0],
            "bit_1_acc": bitwise_accuracies[1],
            "bit_2_acc": bitwise_accuracies[2],
        })

        is_self_condition = torch.randint(2, size=(1,)).item() if ((self.self_cond_mode is not None) and (self.self_cond_mode != "none")) else 0
        if is_self_condition:
            model_output_cont = model_output_cont.detach()
            model_output_disc = model_output_disc.detach()
            null_cond_loss_cont = self._loss(model_output_cont, target_cont, mask)
            null_cond_loss_disc = self._loss(model_output_disc, target_disc, mask)
            if self.discrete_prediction == "x0":
                model_output_disc = torch.clamp(model_output_disc, -1, 1)
            if self.self_cond_mode == "model_output":
                x_input_cont = torch.concat((x_t_cont, model_output_cont), dim = -1)
                x_input_disc = torch.concat((x_t_disc, model_output_disc), dim = -1)
            elif self.self_cond_mode == "x0":
                # discrete predictions are already in x_0, so only process continuous outputs
                # TODO replace denominator with self._sqrt_alpha_bar, instead of doing the math again??
                x_0_cont = (x_t_cont - self._sqrt_alpha_bar_complement[t-1].view(*coeff_shape) * model_output_cont)/torch.sqrt(1 - self._sqrt_alpha_bar_complement[t-1]**2).view(*coeff_shape)
                x_0_cont = torch.clamp(x_0_cont, -3, 3) # avoid nans
                x_0_cont = torch.where(mask, x_cont, x_0_cont) if mask is not None else x_0_cont # mask out ports

                x_input_cont = torch.concat((x_t_cont, x_0_cont), dim = -1)
                x_input_disc = torch.concat((x_t_disc, model_output_disc), dim = -1)

            model_output_cont, model_output_disc = self(x_input_cont, x_input_disc, cond, t)
            
            self_cond_loss_cont = self._loss(model_output_cont, target_cont, mask)
            self_cond_loss_disc = self._loss(model_output_disc, target_disc, mask)
            metrics["self_cond_loss"] = (self_cond_loss_cont + self_cond_loss_disc).detach().cpu().numpy()
            train_loss_cont = self_cond_loss_cont
            train_loss_disc = self_cond_loss_disc
        else:
            null_cond_loss_cont = self._loss(model_output_cont, target_cont, mask)
            null_cond_loss_disc = self._loss(model_output_disc, target_disc, mask)
            train_loss_cont = null_cond_loss_cont
            train_loss_disc = null_cond_loss_disc
        
        train_loss = train_loss_cont + train_loss_disc
        null_cond_loss = null_cond_loss_cont + null_cond_loss_disc
        output_masked_cont = model_output_cont.detach()[torch.logical_not(mask).expand(model_output_cont.shape)] if mask is not None else model_output_cont.detach()
        output_masked_disc = model_output_disc.detach()[torch.logical_not(mask).expand(model_output_disc.shape)] if mask is not None else model_output_disc.detach()
        
        metrics.update({
            "epsilon_theta_mean": output_masked_cont.mean().cpu().numpy(), 
            "epsilon_theta_std": output_masked_cont.std().cpu().numpy(),
            "null_cond_loss": null_cond_loss.detach().cpu().numpy(),
            "loss_continuous": train_loss_cont.detach().cpu().numpy(),
            "loss_discrete": train_loss_disc.detach().cpu().numpy(), 
            })
        
        if self.discrete_prediction == "x0":
            output_masked_disc = torch.abs(output_masked_disc)
            metrics.update({
                "x0_theta_abs_mean": output_masked_disc.mean().cpu().numpy(),
                "x0_theta_abs_std": output_masked_disc.std().cpu().numpy(),
            })
        else:
            metrics.update({
                "epsilon_discrete_mean": output_masked_disc.mean().cpu().numpy(),
                "epsilon_discrete_std": output_masked_disc.std().cpu().numpy(),
            })
        
        return train_loss, metrics
    
    def forward_samples(self, x_cont, x_disc, cond, intermediate_every = 0):
        """
        Simulate forward sampling process for debugging purposes.
        Note that intermediates are NOT discretized
        """
        intermediates_cont = [x_cont]
        intermediates_disc = [x_disc]
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(cond)
        for t in range(self.max_diffusion_steps):
            epsilon_cont = self._epsilon_dist.sample(x_cont.shape).squeeze(dim = -1)
            epsilon_disc = self._epsilon_dist.sample(x_disc.shape).squeeze(dim = -1)
            x_t_cont = self._sqrt_alpha_bar[t] * x_cont + self._sqrt_alpha_bar_complement[t] * epsilon_cont
            x_t_disc = self._sqrt_alpha_bar[t] * x_disc + self._sqrt_alpha_bar_complement[t] * epsilon_disc
            if mask is not None: # don't perturb things covered by mask
                x_t_cont = torch.where(mask, x_cont, x_t_cont)
                x_t_disc = torch.where(mask, x_disc, x_t_disc)
            if intermediate_every and t<(self.max_diffusion_steps-1) and t % intermediate_every == 0:
                intermediates_cont.append(x_t_cont)
                intermediates_disc.append(x_t_disc)
        intermediates_cont.append(x_t_cont) # append final image, non-discretized
        intermediates_disc.append(x_t_disc) # append final image, non-discretized
        return intermediates_cont, intermediates_disc

    def reverse_samples(self, B, x_in_cont, x_in_disc, cond, intermediate_every = 0, mask_override = None):
        """
        x_in_cont and x_in_disc specify positions and orientations of ports
        B: batch size
        intermediate_every: determines how often intermediate diffusion steps are saved and returned. 0 = no intermediates returned
        Note that intermediates are NOT discretized
        """
        batch_shape = (B, cond.x.shape[0], self.input_cont_dim + self.input_disc_dim)
        mask_shape = (1, x_in_cont.shape[1], 1)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(cond)

        x_cont, x_disc = self.split_inputs(self._epsilon_dist.sample(batch_shape).squeeze(dim = -1)) # (B, V, F)
        x_cont = torch.where(mask, x_in_cont, x_cont) if mask is not None else x_cont
        x_disc = torch.where(mask, x_in_disc, x_disc) if mask is not None else x_disc

        self_condition_cont = torch.zeros_like(x_cont)
        self_condition_disc = torch.zeros_like(x_disc)

        intermediates_cont = [x_cont]
        intermediates_disc = [x_disc]
        for t in range(self.max_diffusion_steps, 0 , -1):
            t_vec = torch.tensor(t, device=x_cont.device).expand(B)
            z_combined = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if t>1 else torch.zeros(batch_shape, device=x_cont.device)
            z_cont, z_disc = self.split_inputs(z_combined)
            
            x_input_cont = torch.concat((x_cont, self_condition_cont), dim = -1)
            x_input_disc = torch.concat((x_disc, self_condition_disc), dim = -1)
            
            # compute self-conditioning inputs
            model_output_cont, model_output_disc = self(x_input_cont, x_input_disc, cond, t_vec)
            if self.discrete_prediction == "x0": # clamp outputs
                model_output_disc = torch.clamp(model_output_disc, -1, 1)
            if self.self_cond_mode == "model_output":
                self_condition_cont = model_output_cont
                self_condition_disc = model_output_disc
            elif self.self_cond_mode == "x0":
                # TODO replace denominator with self._sqrt_alpha_bar, instead of doing the math again??
                self_condition_cont = (x_cont.detach() - self._sqrt_alpha_bar_complement[t-1] * model_output_cont.detach())/torch.sqrt(1 - self._sqrt_alpha_bar_complement[t-1]**2)
                self_condition_cont = torch.clamp(self_condition_cont, -3, 3) # avoid nans
                self_condition_cont = torch.where(mask, x_cont, self_condition_cont) if mask is not None else self_condition_cont # mask out ports
                self_condition_disc = model_output_disc
            elif (self.self_cond_mode is None) or (self.self_cond_mode == "none"):
                # self cond inputs stay 0
                pass 
            else:
                raise NotImplementedError
            
            # Do sampling step, given the heterogeneous model outputs (epsilon for continuous, x0 for discrete)
            x_next_cont = (1.0/torch.sqrt(1 - self._beta[t-1])) * (x_cont - (self._beta[t-1]/self._sqrt_alpha_bar_complement[t-1]) * model_output_cont)
            if self.discrete_prediction == "x0":
                epsilon_pred_disc = (x_disc - self._sqrt_alpha_bar[t-1] * model_output_disc)/self._sqrt_alpha_bar_complement[t-1]
            else:
                epsilon_pred_disc = model_output_disc
            x_next_disc = (1.0/torch.sqrt(1 - self._beta[t-1])) * (x_disc - (self._beta[t-1]/self._sqrt_alpha_bar_complement[t-1]) * epsilon_pred_disc)
            # don't denoise things covered by mask
            x_cont = torch.where(mask, x_cont, x_next_cont) if mask is not None else x_next_cont
            x_disc = torch.where(mask, x_disc, x_next_disc) if mask is not None else x_next_disc
            if intermediate_every and t>1 and t % intermediate_every == 0: # TODO output x0 instead
                intermediates_cont.append(x_cont)
                intermediates_disc.append(x_disc)
            x_perturbed_cont = x_cont + self._sigma[t-1] * z_cont
            x_perturbed_disc = x_disc + self._sigma[t-1] * z_disc
            # don't perturb things covered by mask
            x_cont = torch.where(mask, x_cont, x_perturbed_cont) if mask is not None else x_perturbed_cont
            x_disc = torch.where(mask, x_disc, x_perturbed_disc) if mask is not None else x_perturbed_disc
        
        intermediates_cont.append(x_cont) # append final image
        intermediates_disc.append(x_disc) # append final image
        return x_cont, x_disc, intermediates_cont, intermediates_disc
    
    def compute_t_encodings(self, t):
        # t has shape (B,)
        if self.t_encoding_dim == 0:
            return None
        B = t.shape[0]
        encoding = self.t_encoding[t-1, :].view(B, self.t_encoding_dim)
        return encoding

    def combine_inputs(self, x_cont, x_disc):
        """
        Takes continuous and discrete inputs and concats them
        """
        x_combined = torch.cat((x_cont, x_disc), dim=-1)
        return x_combined

    def split_inputs(self, x_combined):
        """
        Takes combined inputs and splits them into continuous and discrete portions.
        Note that continuous occupies the first chunk, disc occupies later indices
        Input: (..., F_cont + F_disc)
        Output: x_cond (..., F_cont), x_disc (..., F_disc)
        """
        x_cont, x_disc = torch.split(x_combined, [self.input_cont_dim, self.input_disc_dim], dim=-1)
        return x_cont, x_disc

    def get_mask(self, cond):
        if self.mask_key and self.mask_key in cond:
            mask = cond[self.mask_key]
            assert len(mask.shape) == 1, "mask must be 1D boolean tensor"
            mask = mask.view(1, -1, 1)
            return mask
        else:
            return None
        
    def _loss(self, x, target, mask = None, debug = False):
        if mask is not None:
            numel = torch.numel(mask)
            squared_error = torch.square(x-target)
            squared_error.masked_fill_(mask, 0)
            if debug:
                mse = torch.sum(squared_error.sum(dim=1), dim=0) / (x.shape[0] * (numel - torch.sum(mask)))
                return mse
            mse = torch.mean(squared_error) * (numel / (numel - torch.sum(mask)))
            return mse
        else:
            return self._lossfn(x, target)

class ChipDiffusionModel(MixedDiffusionModel):
    """
    Wrapper for mixed-diffusion model that provides a nicer interface for model
    """
    def loss(self, x, cond, t):
        """
        Generates orientable placement from fixed input (x, cond)
        """
        orientation, cond_orientable = orientations.to_orientable(cond, randomize=True)
        orientation = orientation.unsqueeze(dim=0).expand(x.shape[0], -1, -1) # (B, V, F)
        return super().loss(x, orientation, cond_orientable, t)
    
    def forward_samples(self, x, cond, intermediate_every=0):
        # TODO fix and discretize outputs
        orientation, cond_orientable = orientations.to_orientable(cond, randomize=False)
        orientation = orientation.unsqueeze(dim=0).expand(x.shape[0], -1, -1) # (B, V, F)
        intermediates_cont, intermediates_disc = super().forward_samples(x, orientation, cond_orientable, intermediate_every=intermediate_every)

        # fix and discretize outputs
        intermediates = [
            torch.cat((cont, orientations.relative_orientation(orientation, disc)), dim=-1) 
            for cont, disc in zip(intermediates_cont, intermediates_disc)
        ]
        return intermediates # these samples are relative to original, fixed cond

    def reverse_samples(self, B, x, cond, intermediate_every=0, mask_override=None):
        orientation, cond_orientable = orientations.to_orientable(cond, randomize=False)
        orientation = orientation.unsqueeze(dim=0).expand(x.shape[0], -1, -1) # (B, V, F)
        x_cont, x_disc, intermediates_cont, intermediates_disc = super().reverse_samples(B, x, orientation, cond_orientable, intermediate_every=intermediate_every, mask_override=mask_override)
        
        # fix and discretize outputs
        output_orientation = orientations.relative_orientation(orientation, x_disc)
        intermediates = [
            torch.cat((cont, orientations.relative_orientation(orientation, disc)), dim=-1) 
            for cont, disc in zip(intermediates_cont, intermediates_disc)
        ]
        output_samples = torch.cat((x_cont, output_orientation), dim=-1) # these samples are relative to original, fixed cond
        return output_samples, intermediates

class NoModel(GuidedDiffusionModel):
    def __init__(
            self,
            **kwargs
        ):
        super().__init__(**kwargs)
        # delete denoising model
        class NoOp(nn.Module):
            def __call__(self, x, cond, t_embed):
                return torch.zeros_like(x)
        self._reverse_model = NoOp()

class ContinuousDiffusionModel(nn.Module):
    backbones = {
        "mlp": networks.ConditionalMLP, 
        "res_mlp": networks.ResidualMLP, 
        "unet": networks.UNet, 
        "cond_unet": networks.CondUNet, 
        "vit": networks.ViT, 
        "res_gnn": networks.ResGNN, 
        "res_gnn_block": networks.ResGNNBlock, 
        "graph_unet": networks.GraphUNet,
        "att_gnn": networks.AttGNN, # Current best
        "graph_transformer": networks.GraphTransformer,
        }
    time_encoders = {
        "sinusoid": pos_encoding.SinusoidContEncoding,
        }
    def __init__(
            self, 
            backbone, 
            backbone_params, 
            input_shape, 
            t_encoding_type, 
            t_encoding_dim, 
            legality_guidance_weight = 0.0,
            hpwl_guidance_weight = 0.0,
            grad_descent_steps = 0,
            grad_descent_rate = 0.0,
            alpha_init = 0.0, # for opt guidance
            alpha_lr = 0.0, # for opt guidance
            alpha_critical_factor = 0.0, # for opt guidance
            legality_potential_target = 0.0, # for opt guidance
            use_adam = False, # for opt guidance
            max_diffusion_steps = 1000, # default eval timesteps
            guidance_mode = "none", # none | sgd (sgd reverse guidance) | opt (constrained opt with adam)
            noise_schedule = "linear", 
            mask_key = None, 
            use_mask_as_input = False, 
            device = "cpu", 
            legality_softmax_factor_min = 10.0,
            legality_softmax_factor_max = 10.0,
            legality_softmax_critical_factor = 0, # fraction of time at which legality factor reaches max
            **kwargs,
            ):
        super().__init__()
        
        with open_dict(backbone_params):
            backbone_params.update({
                "in_node_features": input_shape[1],
                "out_node_features": input_shape[1],
                "t_encoding_dim": t_encoding_dim,
                "device": device,
                "mask_key": mask_key if use_mask_as_input else None,
            })
        
        if t_encoding_dim > 0:
            self.t_encoder = ContinuousDiffusionModel.time_encoders[t_encoding_type](t_encoding_dim)
        self.mask_key = mask_key
        self.t_encoding_dim = t_encoding_dim
        self._reverse_model = ContinuousDiffusionModel.backbones[backbone](**backbone_params)
        self.input_shape = input_shape
        self.max_diffusion_steps = max_diffusion_steps
        if noise_schedule == "cosine":
            self._noise_scheduler = schedulers.CosineScheduler()
        else:
            raise NotImplementedError
        
        self._lossfn = nn.MSELoss(reduction = "mean")

        # cache some variables:
        self.device = device
        self._epsilon_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self._t_dist = tfd.Uniform(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        
        # guidance parameters
        self.is_guided_sampling = \
            (legality_guidance_weight > 0.0 or hpwl_guidance_weight > 0.0 or alpha_lr > 0.0 or alpha_init > 0.0) \
            and (grad_descent_steps > 0) \
            and (grad_descent_rate > 0.0) \
            and (guidance_mode != "none")
        self.guidance_mode = guidance_mode
        self.legality_guidance_weight = legality_guidance_weight
        self.hpwl_guidance_weight = hpwl_guidance_weight
        self.grad_descent_steps = grad_descent_steps # denoted m in universal guidance paper
        self.grad_descent_rate = grad_descent_rate
        # opt guidance parameters
        self.alpha_init = alpha_init
        self.alpha_lr = alpha_lr
        self.alpha_critical_factor = alpha_critical_factor
        self.legality_potential_target = legality_potential_target
        self.use_adam = use_adam
        # legality softmax factor scheduling for guidance
        self.legality_softmax_factor_min = legality_softmax_factor_min
        self.legality_softmax_factor_max = legality_softmax_factor_max
        self.legality_softmax_critical_factor = legality_softmax_critical_factor

        # SVDD-PM parameters (inference-time value-based decoding)
        self.svdd_num_candidates = kwargs.get("svdd_num_candidates", 4)
        self.svdd_alpha_temp = kwargs.get("svdd_alpha_temp", 1.0)
        self.svdd_lambda_legality = kwargs.get("svdd_lambda_legality", 0.0)
        self.svdd_every_n_steps = kwargs.get("svdd_every_n_steps", 5)
        self.svdd_start_step_frac = kwargs.get("svdd_start_step_frac", 0.5)
        # If True, also apply opt's gradient-based guidance to predicted_x0 each reverse step
        self.svdd_layer_opt = kwargs.get("svdd_layer_opt", False)

        # CoDe (blockwise best-of-N) parameters
        self.code_num_candidates = kwargs.get("code_num_candidates", 4)
        self.code_block_size = kwargs.get("code_block_size", 100)
        self.code_lambda_legality = kwargs.get("code_lambda_legality", 1.0)
        self.code_layer_opt = kwargs.get("code_layer_opt", False)

        # TDS (Twisted Diffusion Sampler / SMC) parameters
        self.tds_num_particles = kwargs.get("tds_num_particles", 4)
        self.tds_alpha_temp = kwargs.get("tds_alpha_temp", 1.0)
        self.tds_lambda_legality = kwargs.get("tds_lambda_legality", 1.0)
        self.tds_ess_threshold_frac = kwargs.get("tds_ess_threshold_frac", 0.5)
        self.tds_layer_opt = kwargs.get("tds_layer_opt", False)

    def __call__(self, x, cond, t):
        # input: x is (B, V, F) for graphs, t is (B), cond is Data obj
        # note: 1 graph at a time
        # output: epsilon predictions of model
        t_embed = self.t_encoder(t)
        return self._reverse_model(x, cond, t_embed).view(*x.shape)
    
    def loss(self, x, cond, _, hpwl_weight=0.0, legality_weight=0.0, use_timestep_weighting=False): # last input is for time index; ignored here
        B = x.shape[0] # x is (B, V, F) for graphs
        t = self._t_dist.sample((B,)).squeeze(dim = -1)
        assert t.shape == (B,), "t has to have shape (B,)"

        # prepare mask
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)

        # sample epsilon and generate noisy images
        epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1) # (B, V, F)
        x_perturbed = self._noise_scheduler.add_noise(x, epsilon, t)
        x_perturbed = torch.where(mask, x, x_perturbed) if mask is not None else x_perturbed

        eps_predict = self(x_perturbed, cond, t)

        pred_masked = eps_predict.detach()[torch.logical_not(mask).expand(x.shape)] if mask is not None else eps_predict.detach()
        metrics = {"epsilon_theta_mean": pred_masked.mean().cpu().numpy(), "epsilon_theta_std": pred_masked.std().cpu().numpy()}
        denoising_loss = self._loss(eps_predict, epsilon, mask)

        # Auxiliary losses on predicted_x0 (direct gradient, ReFL-style)
        if hpwl_weight > 0 or legality_weight > 0:
            input_dims = len(x.shape[1:])
            alpha_t = self._noise_scheduler.alpha(t).view((B, *([1] * input_dims)))
            sigma_t = self._noise_scheduler.sigma(t).view((B, *([1] * input_dims)))
            predicted_x0 = (x_perturbed - sigma_t * eps_predict) / alpha_t
            predicted_x0 = torch.where(mask, x, predicted_x0) if mask is not None else predicted_x0
            predicted_x0 = torch.clamp(predicted_x0, -2, 2)

            # Timestep weighting: down-weight early (noisy) timesteps
            if use_timestep_weighting:
                ts_weight = self._noise_scheduler.alpha(t) ** 2  # (B,), in [0, 1]
            else:
                ts_weight = torch.ones_like(t)  # (B,)

            total_loss = denoising_loss
            if hpwl_weight > 0:
                hpwl_per_sample = guidance.hpwl_guidance_potential(predicted_x0, cond)  # (B,)
                hpwl_loss = (ts_weight * hpwl_per_sample).mean()
                total_loss = total_loss + hpwl_weight * hpwl_loss
                metrics["hpwl_loss"] = hpwl_loss.detach().cpu().item()
            if legality_weight > 0:
                legality_per_sample = guidance.legality_guidance_potential(predicted_x0, cond, mask=mask)  # (B,)
                legality_loss = (ts_weight * legality_per_sample).mean()
                total_loss = total_loss + legality_weight * legality_loss
                metrics["legality_loss"] = legality_loss.detach().cpu().item()
            return total_loss, metrics

        return denoising_loss, metrics

    def forward_samples(self, x, cond, intermediate_every = 0):
        intermediates = []
        # prepare mask
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)
        
        step_size = intermediate_every/self.max_diffusion_steps if intermediate_every else 1
        for t in torch.arange(0, 1+(1e-9), step_size):
            epsilon = self._epsilon_dist.sample(x.shape).squeeze(dim = -1) # (B, V, F)
            x_perturbed = self._noise_scheduler.add_noise(x, epsilon, torch.tensor([t], device=x.device))
            x_perturbed = torch.where(mask, x, x_perturbed) if mask is not None else x_perturbed
            intermediates.append(x_perturbed)
        
        return intermediates

    def reverse_samples(self, B, x_in, cond, num_timesteps=-1, intermediate_every = 0, mask_override = None, output_log_prob = False):
        # B: batch size
        # intermediate_every: determines how often intermediate diffusion steps are saved and returned. 0 = no intermediates returned
        if self.guidance_mode == "svdd" and not output_log_prob:
            return self._reverse_samples_svdd(B, x_in, cond, num_timesteps, intermediate_every, mask_override)
        if self.guidance_mode == "code" and not output_log_prob:
            return self._reverse_samples_code(B, x_in, cond, num_timesteps, intermediate_every, mask_override)
        if self.guidance_mode == "tds" and not output_log_prob:
            return self._reverse_samples_tds(B, x_in, cond, num_timesteps, intermediate_every, mask_override)

        batch_shape = (B, cond.x.shape[0], self.input_shape[1])
        mask_shape = (1, x_in.shape[1], 1)

        if num_timesteps <= 0:
            num_timesteps = self.max_diffusion_steps

        x = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) # (B, C, H, W)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        # assert mask is None or mask.sum() == 0, "masks not yet supported when sampling"
        x = torch.where(mask, x_in, x) if mask is not None else x

        intermediates = [x]
        self._noise_scheduler.set_timesteps(num_timesteps)
        timesteps = self._noise_scheduler.timesteps

        if output_log_prob:
            log_probs = torch.zeros((len(timesteps) - 1, B,), device = x.device)
            predicted_x0_list = []

        # reset guidance state (if exists)
        self.reset_guidance_state(dtype = x.dtype)

        for i, (t, t_minus_one) in enumerate(zip(timesteps[:-1], timesteps[1:])):

            if self.is_guided_sampling:
                if self.guidance_mode == "sgd":
                    guidance_fn = lambda x_hat: self.reverse_guidance_force(x_hat, cond, t, mask)
                elif self.guidance_mode == "opt":
                    guidance_fn = lambda x_hat: self.reverse_guidance_opt_force(x_hat, cond, t, mask)
                else:
                    raise NotImplementedError
            else:
                guidance_fn = None

            t_vec = torch.tensor(t, device=x.device).expand(B)
            eps_predict = self(x, cond, t_vec)

            z = self._epsilon_dist.sample(batch_shape).squeeze(dim = -1) if i<(len(timesteps)-2) else torch.zeros_like(x)

            if output_log_prob:
                # Compute mu (mean) and eta (noise scale) to get log probability
                alpha_t = self._noise_scheduler.alpha(t)
                alpha_t_minus_one = self._noise_scheduler.alpha(t_minus_one)
                sigma_t = self._noise_scheduler.sigma(t)
                sigma_t_minus_one = self._noise_scheduler.sigma(t_minus_one)
                eta = self._noise_scheduler.eta(t, t_minus_one)

                predicted_x0 = (x - sigma_t * eps_predict) / alpha_t
                predicted_x0 = torch.where(mask, x, predicted_x0) if mask is not None else predicted_x0
                predicted_x0_list.append(predicted_x0.detach())
                x0_guided = predicted_x0 + (guidance_fn(predicted_x0) if guidance_fn is not None else 0)
                xt_minus_one_direction = torch.sqrt(torch.clip(torch.square(sigma_t_minus_one) - torch.square(eta), min=0)) * eps_predict
                mu = alpha_t_minus_one * x0_guided + xt_minus_one_direction

                x = mu.detach() + eta * z
                x = torch.where(mask, x_in, x) if mask is not None else x
                x = torch.clamp(x, -2, 2)
                log_probs[i, :] = pi_log_prob(x, mu, eta)
            else:
                x, x0_predicted = self._noise_scheduler.step(
                    eps_prediction = eps_predict,
                    t = t,
                    t_minus_one = t_minus_one,
                    xt = x,
                    z = z,
                    mask = mask,
                    guidance_fn = guidance_fn,
                )
                x = torch.clamp(x, -2, 2)

            if intermediate_every and (i+1) % intermediate_every == 0:
                intermediates.append(x)

        if output_log_prob:
            return x, intermediates, log_probs, predicted_x0_list
        else:
            return x, intermediates

    @torch.no_grad()
    def _reverse_samples_svdd(self, B, x_in, cond, num_timesteps=-1, intermediate_every=0, mask_override=None):
        """SVDD-PM (Soft Value-Based Decoding, Posterior-Mean variant).
        Per reverse step (subject to start_step_frac and every_n_steps): draw K candidates,
        score each by reward on its t-1 posterior mean, softmax-resample one."""
        K = int(self.svdd_num_candidates)
        alpha_temp = float(self.svdd_alpha_temp)
        lambda_legality = float(self.svdd_lambda_legality)
        every_n = max(1, int(self.svdd_every_n_steps))
        start_step_frac = float(self.svdd_start_step_frac)
        layer_opt = bool(getattr(self, "svdd_layer_opt", False)) and (self.grad_descent_steps > 0)

        batch_shape = (B, cond.x.shape[0], self.input_shape[1])
        mask_shape = (1, x_in.shape[1], 1)
        if num_timesteps <= 0:
            num_timesteps = self.max_diffusion_steps

        x = self._epsilon_dist.sample(batch_shape).squeeze(dim=-1)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x

        intermediates = [x]
        self._noise_scheduler.set_timesteps(num_timesteps)
        timesteps = self._noise_scheduler.timesteps

        pin_map, pin_offsets, pin_edge_index = guidance.compute_pin_map(cond)
        hpwl_net = guidance.HPWL()

        # If layering opt guidance, initialize opt's stateful alpha/optimizer
        if layer_opt:
            self.reset_guidance_state(dtype=x.dtype)

        for i, (t, t_minus_one) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.tensor(t, device=x.device).expand(B)
            eps_predict = self(x, cond, t_vec)

            alpha_t = self._noise_scheduler.alpha(t)
            alpha_t_minus_one = self._noise_scheduler.alpha(t_minus_one)
            sigma_t = self._noise_scheduler.sigma(t)
            sigma_t_minus_one = self._noise_scheduler.sigma(t_minus_one)
            eta = self._noise_scheduler.eta(t, t_minus_one)

            predicted_x0 = (x - sigma_t * eps_predict) / alpha_t
            if mask is not None:
                predicted_x0 = torch.where(mask, x, predicted_x0)

            # Layered opt guidance: adjust predicted_x0 via paper's gradient-based force
            if layer_opt:
                g = self.reverse_guidance_opt_force(predicted_x0, cond, t, mask)
                predicted_x0 = predicted_x0 + g
                if mask is not None:
                    predicted_x0 = torch.where(mask, x, predicted_x0)

            direction = torch.sqrt(torch.clip(torch.square(sigma_t_minus_one) - torch.square(eta), min=0)) * eps_predict
            mu = alpha_t_minus_one * predicted_x0 + direction

            is_last_step = (i == len(timesteps) - 2)
            do_svdd = (not is_last_step) and (float(t) <= start_step_frac) and (i % every_n == 0)

            if do_svdd:
                # Draw K candidates: K different z's, same mu
                z_cand = self._epsilon_dist.sample((K, *batch_shape)).squeeze(dim=-1)  # (K,B,V,D)
                z_cand = z_cand.reshape(K * B, *batch_shape[1:])
                mu_exp = mu.unsqueeze(0).expand(K, *mu.shape).reshape(K * B, *mu.shape[1:])
                x_cand = mu_exp + eta * z_cand  # (K*B, V, D)
                if mask is not None:
                    x_in_exp = x_in.unsqueeze(0).expand(K, *x_in.shape).reshape(K * B, *x_in.shape[1:])
                    x_cand = torch.where(mask, x_in_exp, x_cand)
                x_cand = torch.clamp(x_cand, -2, 2)

                # Posterior-mean rollout at t-1: one eps_theta forward, Tweedie
                t_minus_one_vec = torch.tensor(t_minus_one, device=x.device).expand(K * B)
                eps_k = self(x_cand, cond, t_minus_one_vec)
                pred_x0_k = (x_cand - sigma_t_minus_one * eps_k) / alpha_t_minus_one
                if mask is not None:
                    pred_x0_k = torch.where(mask, x_cand, pred_x0_k)
                pred_x0_k = torch.clamp(pred_x0_k, -2, 2)

                # Reward = -(HPWL + lambda * legality) on posterior-mean
                hpwl_val = hpwl_net(pred_x0_k, pin_map, pin_offsets, pin_edge_index)  # (K*B,)
                if lambda_legality > 0:
                    leg_val = guidance.legality_guidance_potential(pred_x0_k, cond, mask=mask)  # (K*B,)
                    reward = -(hpwl_val + lambda_legality * leg_val)
                else:
                    reward = -hpwl_val
                reward = reward.view(K, B)

                # Numerical stability: nan/inf -> very negative; per-batch max-subtraction
                reward = torch.where(torch.isfinite(reward), reward, torch.full_like(reward, -1e9))
                reward = reward - reward.max(dim=0, keepdim=True).values.detach()
                weights = torch.softmax(reward / max(alpha_temp, 1e-6), dim=0)  # (K, B)
                weights = torch.nan_to_num(weights, nan=1.0 / K, posinf=1.0, neginf=0.0)
                weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-12)
                idx = torch.multinomial(weights.transpose(0, 1), num_samples=1).squeeze(-1)  # (B,)
                x_kb = x_cand.view(K, B, *x_cand.shape[1:])
                batch_idx = torch.arange(B, device=x.device)
                x = x_kb[idx, batch_idx]
            else:
                z = self._epsilon_dist.sample(batch_shape).squeeze(dim=-1) if not is_last_step else torch.zeros_like(x)
                x = mu + eta * z
                if mask is not None:
                    x = torch.where(mask, x_in, x)
                x = torch.clamp(x, -2, 2)

            if intermediate_every and (i + 1) % intermediate_every == 0:
                intermediates.append(x)

        return x, intermediates

    @torch.no_grad()
    def _reverse_samples_code(self, B, x_in, cond, num_timesteps=-1, intermediate_every=0, mask_override=None):
        """CoDe (Blockwise Best-of-N) sampler. arXiv:2502.00968.
        Every `code_block_size` reverse steps: fork K candidates from current mu,
        evaluate reward on each candidate's t-1 posterior mean, hard-argmax-select one.
        Cheaper than SVDD (K-particle eval only at block boundaries, not every step).
        """
        K = int(self.code_num_candidates)
        block_size = max(1, int(self.code_block_size))
        lambda_legality = float(self.code_lambda_legality)
        layer_opt = bool(getattr(self, "code_layer_opt", False)) and (self.grad_descent_steps > 0)

        batch_shape = (B, cond.x.shape[0], self.input_shape[1])
        mask_shape = (1, x_in.shape[1], 1)
        if num_timesteps <= 0:
            num_timesteps = self.max_diffusion_steps

        x = self._epsilon_dist.sample(batch_shape).squeeze(dim=-1)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x

        intermediates = [x]
        self._noise_scheduler.set_timesteps(num_timesteps)
        timesteps = self._noise_scheduler.timesteps

        pin_map, pin_offsets, pin_edge_index = guidance.compute_pin_map(cond)
        hpwl_net = guidance.HPWL()

        if layer_opt:
            self.reset_guidance_state(dtype=x.dtype)

        for i, (t, t_minus_one) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.tensor(t, device=x.device).expand(B)
            eps_predict = self(x, cond, t_vec)

            alpha_t = self._noise_scheduler.alpha(t)
            alpha_t_minus_one = self._noise_scheduler.alpha(t_minus_one)
            sigma_t = self._noise_scheduler.sigma(t)
            sigma_t_minus_one = self._noise_scheduler.sigma(t_minus_one)
            eta = self._noise_scheduler.eta(t, t_minus_one)

            predicted_x0 = (x - sigma_t * eps_predict) / alpha_t
            if mask is not None:
                predicted_x0 = torch.where(mask, x, predicted_x0)

            if layer_opt:
                g = self.reverse_guidance_opt_force(predicted_x0, cond, t, mask)
                predicted_x0 = predicted_x0 + g
                if mask is not None:
                    predicted_x0 = torch.where(mask, x, predicted_x0)

            direction = torch.sqrt(torch.clip(torch.square(sigma_t_minus_one) - torch.square(eta), min=0)) * eps_predict
            mu = alpha_t_minus_one * predicted_x0 + direction

            is_last_step = (i == len(timesteps) - 2)
            # CoDe: only do best-of-N at block boundaries (every block_size steps, but not step 0)
            do_code = (not is_last_step) and (i > 0) and (i % block_size == 0)

            if do_code:
                z_cand = self._epsilon_dist.sample((K, *batch_shape)).squeeze(dim=-1)
                z_cand = z_cand.reshape(K * B, *batch_shape[1:])
                mu_exp = mu.unsqueeze(0).expand(K, *mu.shape).reshape(K * B, *mu.shape[1:])
                x_cand = mu_exp + eta * z_cand
                if mask is not None:
                    x_in_exp = x_in.unsqueeze(0).expand(K, *x_in.shape).reshape(K * B, *x_in.shape[1:])
                    x_cand = torch.where(mask, x_in_exp, x_cand)
                x_cand = torch.clamp(x_cand, -2, 2)

                t_minus_one_vec = torch.tensor(t_minus_one, device=x.device).expand(K * B)
                eps_k = self(x_cand, cond, t_minus_one_vec)
                pred_x0_k = (x_cand - sigma_t_minus_one * eps_k) / alpha_t_minus_one
                if mask is not None:
                    pred_x0_k = torch.where(mask, x_cand, pred_x0_k)
                pred_x0_k = torch.clamp(pred_x0_k, -2, 2)

                hpwl_val = hpwl_net(pred_x0_k, pin_map, pin_offsets, pin_edge_index)
                if lambda_legality > 0:
                    leg_val = guidance.legality_guidance_potential(pred_x0_k, cond, mask=mask)
                    reward = -(hpwl_val + lambda_legality * leg_val)
                else:
                    reward = -hpwl_val
                reward = reward.view(K, B)

                # Hard argmax over K (CoDe's defining feature vs SVDD's softmax)
                reward = torch.where(torch.isfinite(reward), reward, torch.full_like(reward, -1e9))
                idx = reward.argmax(dim=0)  # (B,)
                x_kb = x_cand.view(K, B, *x_cand.shape[1:])
                batch_idx = torch.arange(B, device=x.device)
                x = x_kb[idx, batch_idx]
            else:
                z = self._epsilon_dist.sample(batch_shape).squeeze(dim=-1) if not is_last_step else torch.zeros_like(x)
                x = mu + eta * z
                if mask is not None:
                    x = torch.where(mask, x_in, x)
                x = torch.clamp(x, -2, 2)

            if intermediate_every and (i + 1) % intermediate_every == 0:
                intermediates.append(x)

        return x, intermediates

    @torch.no_grad()
    def _reverse_samples_tds(self, B, x_in, cond, num_timesteps=-1, intermediate_every=0, mask_override=None):
        """TDS (Twisted Diffusion Sampler / SMC). arXiv:2306.17775.
        Maintain N particles running reverse process in parallel.
        Per step: standard reverse, then importance weight update via delta of
        value(predicted_x0) (reward improvement). ESS-based resampling.
        Output: best-final-reward particle (collapses N -> 1)."""
        N = int(self.tds_num_particles)
        alpha_temp = float(self.tds_alpha_temp)
        lambda_legality = float(self.tds_lambda_legality)
        ess_threshold_frac = float(self.tds_ess_threshold_frac)
        layer_opt = bool(getattr(self, "tds_layer_opt", False)) and (self.grad_descent_steps > 0)

        assert B == 1, f"TDS expects B=1 (per-sample particle filter), got B={B}"

        batch_shape = (N, cond.x.shape[0], self.input_shape[1])
        mask_shape = (1, x_in.shape[1], 1)
        if num_timesteps <= 0:
            num_timesteps = self.max_diffusion_steps

        x = self._epsilon_dist.sample(batch_shape).squeeze(dim=-1)  # (N, V, F)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x_in_n = x_in.expand(N, *x_in.shape[1:]) if x_in.shape[0] == 1 else x_in
        x = torch.where(mask, x_in_n, x) if mask is not None else x

        intermediates = [x[:1]]
        self._noise_scheduler.set_timesteps(num_timesteps)
        timesteps = self._noise_scheduler.timesteps

        pin_map, pin_offsets, pin_edge_index = guidance.compute_pin_map(cond)
        hpwl_net = guidance.HPWL()

        if layer_opt:
            self.reset_guidance_state(dtype=x.dtype)

        ess_threshold = ess_threshold_frac * N
        log_w = torch.zeros(N, device=x.device)
        prev_value = None  # value of predicted_x0 at previous step's x

        def _value(x_eval, t_eval):
            """Compute reward = -(HPWL + lambda * legality) on Tweedie predicted_x0(x_eval, t_eval). Returns (N,)."""
            t_v = torch.tensor(t_eval, device=x_eval.device).expand(x_eval.shape[0])
            eps_e = self(x_eval, cond, t_v)
            a_t = self._noise_scheduler.alpha(t_eval)
            s_t = self._noise_scheduler.sigma(t_eval)
            px0 = (x_eval - s_t * eps_e) / a_t
            if mask is not None:
                px0 = torch.where(mask, x_eval, px0)
            px0 = torch.clamp(px0, -2, 2)
            h_v = hpwl_net(px0, pin_map, pin_offsets, pin_edge_index)
            if lambda_legality > 0:
                l_v = guidance.legality_guidance_potential(px0, cond, mask=mask)
                return -(h_v + lambda_legality * l_v)
            return -h_v

        for i, (t, t_minus_one) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # Standard reverse: x_t -> x_{t-1}
            t_vec = torch.tensor(t, device=x.device).expand(N)
            eps_predict = self(x, cond, t_vec)

            alpha_t = self._noise_scheduler.alpha(t)
            alpha_t_minus_one = self._noise_scheduler.alpha(t_minus_one)
            sigma_t = self._noise_scheduler.sigma(t)
            sigma_t_minus_one = self._noise_scheduler.sigma(t_minus_one)
            eta = self._noise_scheduler.eta(t, t_minus_one)

            predicted_x0 = (x - sigma_t * eps_predict) / alpha_t
            if mask is not None:
                predicted_x0 = torch.where(mask, x, predicted_x0)

            if layer_opt:
                g = self.reverse_guidance_opt_force(predicted_x0, cond, t, mask)
                predicted_x0 = predicted_x0 + g
                if mask is not None:
                    predicted_x0 = torch.where(mask, x, predicted_x0)

            direction = torch.sqrt(torch.clip(torch.square(sigma_t_minus_one) - torch.square(eta), min=0)) * eps_predict
            mu = alpha_t_minus_one * predicted_x0 + direction

            is_last_step = (i == len(timesteps) - 2)
            z = self._epsilon_dist.sample(batch_shape).squeeze(dim=-1) if not is_last_step else torch.zeros_like(x)
            x = mu + eta * z
            if mask is not None:
                x = torch.where(mask, x_in_n, x)
            x = torch.clamp(x, -2, 2)

            # Importance weight update via value-delta (skip last step where t≈0)
            if not is_last_step:
                value_now = _value(x, t_minus_one)  # (N,)
                value_now = torch.where(torch.isfinite(value_now), value_now, torch.full_like(value_now, -1e9))
                if prev_value is not None:
                    delta = value_now - prev_value
                    delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
                    log_w = log_w + delta / max(alpha_temp, 1e-6)
                else:
                    log_w = value_now / max(alpha_temp, 1e-6)
                prev_value = value_now

                # ESS check
                log_w_stable = log_w - log_w.max()
                w_norm = torch.softmax(log_w_stable, dim=0)
                w_norm = torch.nan_to_num(w_norm, nan=1.0 / N, posinf=1.0, neginf=0.0)
                w_norm = w_norm / (w_norm.sum() + 1e-12)
                ess = 1.0 / ((w_norm * w_norm).sum() + 1e-12)

                if ess < ess_threshold:
                    idx = torch.multinomial(w_norm, num_samples=N, replacement=True)
                    x = x[idx]
                    prev_value = prev_value[idx]
                    log_w = torch.zeros(N, device=x.device)

            if intermediate_every and (i + 1) % intermediate_every == 0:
                intermediates.append(x[:1])

        # Best-of-N final selection by reward on x (final placement)
        hpwl_f = hpwl_net(x, pin_map, pin_offsets, pin_edge_index)
        if lambda_legality > 0:
            leg_f = guidance.legality_guidance_potential(x, cond, mask=mask)
            final_reward = -(hpwl_f + lambda_legality * leg_f)
        else:
            final_reward = -hpwl_f
        final_reward = torch.where(torch.isfinite(final_reward), final_reward, torch.full_like(final_reward, -1e9))
        best_idx = final_reward.argmax()
        x_out = x[best_idx:best_idx + 1]  # (1, V, F)

        return x_out, intermediates

    def get_mask(self, x, cond):
        if self.mask_key and self.mask_key in cond:
            mask = cond[self.mask_key]
            B, V, F = x.shape
            mask = mask.view(1, V, 1)
            return mask
        else:
            return None

    def _loss(self, prediction, target, mask = None):
        if mask is not None:
            numel = torch.numel(mask)
            squared_error = torch.square(prediction-target)
            squared_error.masked_fill_(mask, 0)
            mse = torch.mean(squared_error) * (numel / (numel - torch.sum(mask)))
            return mse
        else:
            return self._lossfn(prediction, target)
    
    @torch.enable_grad()
    def reverse_guidance_force(self, x_hat, cond, t, mask=None):
        """
        Gives the guidance term for universal backwards guidance. This function is stateless
        Inputs:
        - mask (1, V, 1)
        - x_hat (B, V, 2) is predicted x0
        - t is float from 0 to 1
        Note that the guidance force returned is in x0 space, and is not rescaled
        """
        # TODO Cache pin maps for hpwl guidance
        B, V, D = x_hat.shape
        legality_softmax_factor = self.get_legality_softmax_factor(t)

        # compute backward guidance on x_hat
        x_hat_current = x_hat.detach().clone().requires_grad_(True)
        optimizer = torch.optim.SGD((x_hat_current,), lr=self.grad_descent_rate, momentum=0.0)
        # m step gradient descent
        for _ in range(self.grad_descent_steps):
            optimizer.zero_grad()
            h_legality = self.legality_guidance_weight * guidance.legality_guidance_potential(x_hat_current, cond, mask=mask, softmax_factor=legality_softmax_factor)
            h_hpwl = self.hpwl_guidance_weight * guidance.hpwl_guidance_potential(x_hat_current, cond)
            h = h_legality + h_hpwl
            h.sum().backward()
            optimizer.step()
        
        delta_x_hat = x_hat_current.detach() - x_hat.detach()
        g = delta_x_hat
        return g.detach()
    
    @torch.enable_grad()
    def reverse_guidance_opt_force(self, x_hat, cond, t, mask=None):
        """
        Gives the guidance term for universal backwards guidance using opt. 
        Auto-tunes legality weight using optimization principles (inspired by SAC).
        This function maintains some state (alpha weighting for legality, optimizers).
        Inputs:
        - mask (1, V, 1)
        - x_hat (B, V, 2) is predicted x0
        - t is float from 0 to 1
        Note that the guidance force returned is in x0 space, and is not rescaled
        """
        # TODO Cache pin maps for hpwl guidance
        B, V, D = x_hat.shape
        legality_softmax_factor = self.get_legality_softmax_factor(t)
        
        # compute backward guidance on x_hat
        x_hat_current = x_hat.detach().clone().requires_grad_(True)
        if self.use_adam:
            optimizer_x = torch.optim.Adam((x_hat_current,), lr=self.grad_descent_rate, betas=(0.8, 0.99))
        else:
            optimizer_x = torch.optim.SGD((x_hat_current,), lr=self.grad_descent_rate, momentum=0.0)
        
        # m step gradient descent
        for _ in range(self.grad_descent_steps):

            # gradient step wrt x
            optimizer_x.zero_grad()
            h_legality_raw = guidance.legality_guidance_potential(x_hat_current, cond, mask=mask, softmax_factor=legality_softmax_factor)
            h_legality = self.alpha.detach().item() * h_legality_raw
            h_hpwl = self.hpwl_guidance_weight * guidance.hpwl_guidance_potential(x_hat_current, cond)
            h = h_legality + h_hpwl
            h.sum().backward()
            # Stop ports from moving due to guidance
            if mask is not None:
                x_hat_current.grad *= (~mask).float()
            optimizer_x.step()

            # gradient step wrt alpha
            if t < self.alpha_critical_factor:
                self.optimizer_alpha.zero_grad()
                alpha_cost = -self.alpha * (h_legality_raw.detach() - self.legality_potential_target).mean()
                alpha_cost.backward()
                self.optimizer_alpha.step()
                
                # for numerical stability
                self.alpha.data.clip_(max=15)
        
        delta_x_hat = x_hat_current.detach() - x_hat.detach()
        g = delta_x_hat
        return g.detach()

    def get_legality_softmax_factor(self, t):
        if t > self.legality_softmax_critical_factor:
            t_prime = (t-self.legality_softmax_critical_factor)/(1-self.legality_softmax_critical_factor)
            legality_softmax_factor = self.legality_softmax_factor_max - t_prime * (self.legality_softmax_factor_max-self.legality_softmax_factor_min)
        else:
            return self.legality_softmax_factor_max
        return legality_softmax_factor

    def reset_guidance_state(self, dtype=float):
        # reset alpha, optimizers, etc
        self.alpha = torch.tensor(self.alpha_init, dtype = dtype, device = self.device, requires_grad = True)
        if self.use_adam:
            self.optimizer_alpha = torch.optim.Adam((self.alpha,), lr=self.alpha_lr, betas=(0.9, 0.99))
        else:
            self.optimizer_alpha = torch.optim.SGD((self.alpha,), lr=self.alpha_lr, momentum=0.0)
        return

class FlowMatchingModel(ContinuousDiffusionModel):
    """Conditional Flow Matching (linear / OT path) sibling of ContinuousDiffusionModel.

    Reuses the same backbone, mask handling, time encoder and __init__ kwargs so that
    the existing `model:` config section applies unchanged (only `family=flow_matching`).
    The network output `self(x_t, cond, t)` is interpreted as a velocity field v_theta.

    Convention: x0 = data placement, x1 ~ N(0, I), x_t = (1-t)*x0 + t*x1, and the sampler
    integrates the ODE dx/dt = v backwards from t=1 (noise) to t=0 (data).
    Guidance is intentionally not ported here (Phase A only uses guidance_mode == "none").
    """

    def loss(self, x, cond, _, hpwl_weight=0.0, legality_weight=0.0, use_timestep_weighting=False):
        # last input (t index) ignored; we sample continuous t ~ U(0,1) internally
        B = x.shape[0]  # x is (B, V, F)
        t = self._t_dist.sample((B,)).squeeze(dim=-1)  # (B,) in [0,1]
        assert t.shape == (B,), "t has to have shape (B,)"

        # prepare mask (ports are fixed / excluded from loss)
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)

        # linear/OT interpolation path
        x1 = self._epsilon_dist.sample(x.shape).squeeze(dim=-1)  # (B, V, F) noise
        input_dims = len(x.shape[1:])
        t_b = t.view((B, *([1] * input_dims)))
        x_t = (1.0 - t_b) * x + t_b * x1
        x_t = torch.where(mask, x, x_t) if mask is not None else x_t

        v_target = x1 - x  # target velocity v* = x1 - x0
        v_predict = self(x_t, cond, t)

        pred_masked = v_predict.detach()[torch.logical_not(mask).expand(x.shape)] if mask is not None else v_predict.detach()
        metrics = {"v_theta_mean": pred_masked.mean().cpu().numpy(), "v_theta_std": pred_masked.std().cpu().numpy()}
        loss = self._loss(v_predict, v_target, mask)

        # Auxiliary losses on the estimated clean placement x0_hat = x_t - t * v_theta
        if hpwl_weight > 0 or legality_weight > 0:
            predicted_x0 = x_t - t_b * v_predict
            predicted_x0 = torch.where(mask, x, predicted_x0) if mask is not None else predicted_x0
            predicted_x0 = torch.clamp(predicted_x0, -2, 2)
            ts_weight = (1.0 - t) if use_timestep_weighting else torch.ones_like(t)
            total_loss = loss
            if hpwl_weight > 0:
                hpwl_per_sample = guidance.hpwl_guidance_potential(predicted_x0, cond)
                hpwl_loss = (ts_weight * hpwl_per_sample).mean()
                total_loss = total_loss + hpwl_weight * hpwl_loss
                metrics["hpwl_loss"] = hpwl_loss.detach().cpu().item()
            if legality_weight > 0:
                legality_per_sample = guidance.legality_guidance_potential(predicted_x0, cond, mask=mask)
                legality_loss = (ts_weight * legality_per_sample).mean()
                total_loss = total_loss + legality_weight * legality_loss
                metrics["legality_loss"] = legality_loss.detach().cpu().item()
            return total_loss, metrics

        return loss, metrics

    def forward_samples(self, x, cond, intermediate_every=0):
        # linear interpolation path from data (t=0) toward noise (t=1)
        intermediates = []
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)
        step_size = intermediate_every / self.max_diffusion_steps if intermediate_every else 1
        for t in torch.arange(0, 1 + (1e-9), step_size):
            t_b = t.to(x.device)
            x1 = self._epsilon_dist.sample(x.shape).squeeze(dim=-1)
            x_t = (1.0 - t_b) * x + t_b * x1
            x_t = torch.where(mask, x, x_t) if mask is not None else x_t
            intermediates.append(x_t)
        return intermediates

    def reverse_samples(self, B, x_in, cond, num_timesteps=-1, intermediate_every=0, mask_override=None, output_log_prob=False):
        # Euler ODE integration from t=1 (noise) to t=0 (data): x <- x - dt * v_theta
        batch_shape = (B, cond.x.shape[0], self.input_shape[1])
        mask_shape = (1, x_in.shape[1], 1)

        if num_timesteps <= 0:
            num_timesteps = self.max_diffusion_steps

        x = self._epsilon_dist.sample(batch_shape).squeeze(dim=-1)  # x at t=1 (pure noise)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x

        intermediates = [x]
        # uniform time grid from 1 -> 0 with num_timesteps Euler steps
        ts = torch.linspace(1.0, 0.0, num_timesteps + 1, device=x.device)

        # opt guidance on the x0_hat estimate, same mechanism as the DDPM sampler.
        # On the linear path the guided update is exact: given x0_hat at time t,
        # x_{t_next} = x0_hat + (t_next/t) * (x_t - x0_hat).
        use_guidance = self.is_guided_sampling and self.guidance_mode == "opt"
        if use_guidance:
            self.reset_guidance_state(dtype=x.dtype)

        if output_log_prob:
            log_probs = torch.zeros((num_timesteps, B,), device=x.device)
            predicted_x0_list = []

        for i in range(num_timesteps):
            t = ts[i]
            t_next = ts[i + 1]
            dt = (t - t_next)  # positive step size
            t_vec = t.expand(B)
            v = self(x, cond, t_vec)

            # estimated clean placement x0_hat = x - t * v
            predicted_x0 = x - t * v
            predicted_x0 = torch.where(mask, x_in, predicted_x0) if mask is not None else predicted_x0

            if output_log_prob:
                predicted_x0_list.append(predicted_x0.detach())

            if use_guidance:
                g = self.reverse_guidance_opt_force(predicted_x0, cond, float(t), mask)
                x0_guided = predicted_x0 + g
                x0_guided = torch.where(mask, x_in, x0_guided) if mask is not None else x0_guided
                x = x0_guided + (t_next / t) * (x - x0_guided)
            else:
                x = x - dt * v
            x = torch.where(mask, x_in, x) if mask is not None else x
            x = torch.clamp(x, -2, 2)

            if intermediate_every and (i + 1) % intermediate_every == 0:
                intermediates.append(x)

        if output_log_prob:
            return x, intermediates, log_probs, predicted_x0_list
        else:
            return x, intermediates

def pi_log_prob(x_t_minus, mu, sigma):
    # mu should have gradients, x_t_minus should be detached
    # x_t_minus: (B, ...)
    # mu: (B, ...)
    # sigma: float
    # output: tensor(B) with gradients
    dims = list(range(1, len(mu.shape)))
    return -0.5 * torch.mean(torch.square((x_t_minus.detach() - mu)), dim=dims)/(sigma**2)

def get_linear_sched(T, beta_1, beta_T):
    # returns noise schedule beta as numpy array with shape (T)
    return np.linspace(beta_1, beta_T, T)

def compute_alpha(beta):
    # computes alpha^bar as numpy array with shape (T)
    # input: (T)
    alpha = 1-beta
    alpha_bar = np.multiply.accumulate(alpha)
    return alpha_bar

import torch
import guidance

class DDPO():
    
    def __init__(self, model, reward_fn, batch_size, ema_factor=0.99, warmup_size=200, num_timesteps=-1, clip_epsilon=0.0, local_reward_weight=0.0, local_reward_every=1):
        """
        model: nn.Module with reverse_samples() function that can output log probs
        reward_fn: callable
        batch_size: int
        num_timesteps: number of diffusion steps for reverse sampling (-1 = use model default)
        clip_epsilon: PPO clipping range (0 = no clipping, use REINFORCE)
        local_reward_weight: weight for per-step local reward (0 = disabled)
        local_reward_every: compute local reward every N steps (to save compute)
        """
        self.model = model
        self.reward_fn = reward_fn
        self.local_reward_fn = get_reward_fn(0.0, 1.0)  # local reward = HPWL only (no V×V legality)
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.clip_epsilon = clip_epsilon
        self.local_reward_weight = local_reward_weight
        self.local_reward_every = local_reward_every

        self.ema_factor = ema_factor
        self.warmup_size = warmup_size

        self.rew_buffer = None
        self.ema_mean = None
        self.ema_std = None
    
    def loss(self, x, cond):
        # Note: here x is only used because reverse sampling requires it for port positions
        # Don't need intermediates
        # log_prob is (T, B) tensor with gradients
        extra_kwargs = {"num_timesteps": self.num_timesteps} if self.num_timesteps > 0 else {}

        x_0, _, log_prob, predicted_x0_list = self.model.reverse_samples(self.batch_size, x, cond, intermediate_every = 0, output_log_prob = True, **extra_kwargs)
        x_0 = x_0.detach()
        reward = self.reward_fn(x_0, cond).detach() # (B)

        self.update_moving_averages(reward)
        advantage = (reward - self.ema_mean) / self.ema_std

        trajectory_log_prob = log_prob.mean(dim=0)  # (B,)

        if self.clip_epsilon > 0:
            # Clamp advantage to prevent large updates
            clipped_advantage = torch.clamp(advantage, -1.0 / self.clip_epsilon, 1.0 / self.clip_epsilon)
            # Normalize by log_prob magnitude to keep loss on ~O(1) scale
            log_prob_scale = trajectory_log_prob.detach().abs().mean().clamp(min=1.0)
            global_loss = -torch.mean((trajectory_log_prob / log_prob_scale) * clipped_advantage)
        else:
            # REINFORCE (original)
            global_loss = -torch.mean(trajectory_log_prob * advantage)

        # Local reward: per-step HPWL reward on predicted x0
        local_loss = torch.tensor(0.0, device=x.device)
        if self.local_reward_weight > 0 and len(predicted_x0_list) > 0:
            T = len(predicted_x0_list)
            local_rewards = []
            for t_idx in range(0, T, self.local_reward_every):
                local_r = self.local_reward_fn(predicted_x0_list[t_idx], cond).detach()  # (B,)
                local_rewards.append(local_r)
            # Average local reward across sampled steps
            local_reward_mean = torch.stack(local_rewards).mean(dim=0)  # (B,)
            local_advantage = (local_reward_mean - local_reward_mean.mean()) / local_reward_mean.std().clamp(min=1e-6)
            if self.clip_epsilon > 0:
                local_advantage = torch.clamp(local_advantage, -1.0 / self.clip_epsilon, 1.0 / self.clip_epsilon)
            local_loss = -torch.mean((trajectory_log_prob / log_prob_scale) * local_advantage)

        loss = global_loss + self.local_reward_weight * local_loss

        metrics = {
            "reward": reward.detach().mean().cpu().numpy(),
            "reward_ema_mean": self.ema_mean.cpu().numpy(),
            "reward_ema_std": self.ema_std.cpu().numpy(),
            "local_reward": local_reward_mean.mean().cpu().item() if self.local_reward_weight > 0 else 0.0,
        }
        return loss, metrics
    
    def update_moving_averages(self, new_reward):
        # new reward has shape (B)
        if self.rew_buffer is None:
            self.rew_buffer = new_reward
        buff_len = self.rew_buffer.shape[0]
        if buff_len < self.warmup_size:
            self.rew_buffer = torch.cat((self.rew_buffer, new_reward), dim=0)
            self.ema_mean = self.rew_buffer.mean()
            self.ema_std = self.rew_buffer.std()
        else:
            self.ema_mean = self.ema_factor * self.ema_mean + (1-self.ema_factor) * new_reward.mean()
            self.ema_std = self.ema_factor * self.ema_std + (1-self.ema_factor) * new_reward.std()

def legality_reward(x, cond):
    return -guidance.legality_guidance_potential(x, cond) # better legality means better reward

def hpwl_reward(x, cond):
    return -guidance.hpwl_guidance_potential(x, cond) # lower hpwl means higher reward

def get_reward_fn(legality_weight, hpwl_weight):
    def reward_fn(x, cond):
        return legality_weight * legality_reward(x, cond) + hpwl_weight * hpwl_reward(x, cond)
    return reward_fn
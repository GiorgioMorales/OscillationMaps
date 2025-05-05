import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from OscillationMaps.Models.unet import UNet 


class Network:
    def __init__(self, unet, timesteps, device, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.loss_fn = None
        self.posterior_mean_coef2 = None
        self.posterior_mean_coef1 = None
        self.posterior_log_variance_clipped = None
        self.sqrt_recipm1_gammas = None
        self.gammas = None
        self.sqrt_recip_gammas = None
        self.timesteps = None
        self.denoise_fn = unet
        self.beta_schedule = None
        self.timesteps = timesteps
        self.device = device
        self.max_diff = None
        self.min_diff = None

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_stats(self, max_diff, min_diff):
        self.max_diff = max_diff
        self.min_diff = min_diff

    def compute_channel_scales(self, y_cond, min_scale=1, max_scale=6):
        """Compute per-channel scale based on dataset statistics."""
        per_image_diffs = (y_cond.amax(dim=(-2, -1)) - y_cond.amin(dim=(-2, -1))).view(y_cond.size(0), -1)
        scales = (per_image_diffs - self.min_diff) * (max_scale - min_scale) / (
                    self.max_diff - self.min_diff) + min_scale
        return scales

    def set_new_noise_schedule(self, scales=None):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        # self.beta_schedule = make_beta_schedule('cosine', n_timestep=self.timesteps)
        # self.beta_schedule = self.beta_schedule.detach().cpu().numpy() if isinstance(
        #     self.beta_schedule, torch.Tensor) else self.beta_schedule
        #
        # if scales is not None:
        #     batch_size, num_channels = scales.shape
        #     self.beta_schedule = np.expand_dims(self.beta_schedule, axis=(0, 1))  # Shape: (1, 1, timesteps)
        #     self.beta_schedule = np.repeat(self.beta_schedule, batch_size, axis=0)
        #     self.beta_schedule = np.repeat(self.beta_schedule, num_channels, axis=1)  # Expand to fit "scales" shape
        #     self.beta_schedule *= scales.detach().cpu().numpy()[..., np.newaxis]

        batch_size, num_channels = scales.shape
        self.beta_schedule = np.zeros((batch_size, num_channels, self.timesteps))
        for b in range(batch_size):
            for c in range(num_channels):
                beta_schedule = make_beta_schedule('cosine', n_timestep=self.timesteps, scale=scales[b][c].item())
                self.beta_schedule[b][c] = beta_schedule.detach().cpu().numpy()

        alphas = 1. - self.beta_schedule

        timesteps = self.beta_schedule.shape[-1]
        self.timesteps = int(timesteps)

        if scales is not None:
            gammas = np.cumprod(alphas, axis=-1)  # Compute cumulative product along timesteps
            gammas_prev = np.concatenate([np.ones((scales.shape[0], scales.shape[1], 1)), gammas[..., :-1]], axis=-1)
        else:
            gammas = np.cumprod(alphas, axis=0)
            gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.gammas = to_torch(gammas)
        self.sqrt_recip_gammas = to_torch(np.sqrt(1. / gammas))
        self.sqrt_recipm1_gammas = to_torch(np.sqrt(1. / gammas - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = self.beta_schedule * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = to_torch(self.beta_schedule * np.sqrt(gammas_prev) / (1. - gammas))
        self.posterior_mean_coef2 = to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
                extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
                extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
                extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
                sample_gammas.sqrt().unsqueeze(-1).unsqueeze(-1) * y_0 +
                (1 - sample_gammas).sqrt().unsqueeze(-1).unsqueeze(-1) * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape

        assert self.timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.timesteps // sample_num)

        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None):
        # Calculate scales and new schedule
        scales = self.compute_channel_scales(y_cond=y_cond)
        self.set_new_noise_schedule(scales=scales)
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = 0 * torch.randint(1, self.timesteps, (b,), device=y_0.device).long() + 99  # TODO
        gamma_t1 = self.gammas[torch.arange(self.gammas.size(0))[:, None],
                               torch.arange(self.gammas.size(1))[None, :],
                              (t - 1).unsqueeze(-1)
                              ].squeeze(-1)
        gamma_t2 = self.gammas[torch.arange(self.gammas.size(0))[:, None],
                               torch.arange(self.gammas.size(1))[None, :],
                               t.unsqueeze(-1)
                              ].squeeze(-1)
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = torch.randn_like(y_0) * scales.unsqueeze(-1).unsqueeze(-1) / 6
        # noise = torch.randn_like(y_0)
        y_noisy = self.q_sample(y_0=y_0, sample_gammas=sample_gammas, noise=noise)

        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def cosine_schedule(t, start=0, end=1, tau=1., clip_min=1e-9):
    """
    Compute a gamma function based on the cosine function.
    - `tau > 1`: Slower noise increase (smoother diffusion).
    - `tau < 1`: Faster noise increase.
    """
    v_start = math.cos(start * math.pi / 2) ** tau
    v_end = math.cos(end * math.pi / 2) ** tau
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** tau
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3, scale=1.0):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        # alphas = (timesteps / (1 + cosine_s) * (math.pi / 2)) ** scale
        # alphas = torch.cos(alphas).pow(2)
        # alphas = alphas / alphas[0]
        #
        # betas = 1 - alphas[1:] / alphas[:-1]
        # betas = betas.clamp(max=0.999)
        alphas = torch.tensor([cosine_schedule(t, start=cosine_s, tau=scale) for t in timesteps], dtype=torch.float64)
        alphas = alphas / alphas[0]  # Normalize

        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
        # betas[-1] = 1e-9
    else:
            raise NotImplementedError(schedule)
    return betas
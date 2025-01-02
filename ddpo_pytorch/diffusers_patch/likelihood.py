from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
# reference: Yang Song's SDE tutorial jupyter notebook
# https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3#scrollTo=DfOkg5jBZcjF


def get_prior_likelihood(
    z: torch.FloatTensor, 
    sigma: float,
):
    
    shape = torch.tensor(z.shape)
    N = torch.prod(shape[2:])
    return -N / 2. * np.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(2,3,4)) / (2 * sigma**2)
    


def ode_likelihood(
    pipeline,
    x: Optional[torch.FloatTensor] = None,
    latent: Optional[torch.FloatTensor] = None,
    timestep: int = 1,
    # eps: float = 1e-5, # smallest timestep for numerical stability
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    # num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    num_inference_steps: int = 50,
    solver: str = "euler",
    atol: float = 5e-2,
    rtol: float = 5e-2,
):
    # step 0
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    
    # step 1 
    pipeline.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds, 
        negative_prompt_embeds
    )
    
    # step 2
    # if prompt is not None and isinstance(prompt, str):
    #     batch_size = 1
    # elif prompt is not None and isinstance(prompt, list):
    #     batch_size = len(prompt)
    # else:
    #     batch_size = prompt_embeds.shape[0]
        
    device = pipeline._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    
    # step 3 encode the prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    
    prompt_embeds = pipeline._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )[:prompt_embeds.shape[0], :, :] ## TODO -> 
    # https://github.com/huggingface/diffusers/blob/6427aa995e0c03c1c1a635cd6af8e365f47541a8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L302
    # classifier free guidnace 도입하려면 원래대로 돌려 놔야 한다. negative prompt와 positive prompt가 concat되어 있음
    
    # step 4 : skip the timestep preparation,
    # step 5 : prepare latent variables
    # step 6 : prepare extra step kwawrgs 
    
    if latent is not None:
        encoded_latents = latent
    else:
        shape = x.shape
        x = pipeline.image_processor.preprocess(x, height, width)
        encoded_latents = pipeline.vae.encode( ### TODO stochasticity 제거하기
                x
        ).latent_dist.sample(generator) ###TODO we have to encode the given image and feed it to the pipeline
        encoded_latents = encoded_latents * pipeline.vae.config.scaling_factor
        
    # get the sigma for the DDIM eq (14)
    step_size = int(pipeline.scheduler.config.num_train_timesteps / num_inference_steps)
    sigmas = (((1 - pipeline.scheduler.alphas_cumprod) / pipeline.scheduler.alphas_cumprod) ** 0.5).to(device).to(torch.float32)
    # sigmas = torch.cat([torch.tensor(0).to(device), sigmas]) # 1001
    sampled_sigmas = sigmas[::step_size]
    sampled_sigmas = sampled_sigmas[(sampled_sigmas >= sigmas[timestep - 1])]
    sampled_sigmas = torch.cat([sampled_sigmas, sigmas.max()[None]])
    if timestep == 1:
        sampled_sigmas = torch.cat([torch.tensor([1e-6]).to(device), sampled_sigmas])
        
    epsilon = torch.randint_like(encoded_latents[0], 2) * 2 - 1
    assert sigmas != None
    assert not torch.isnan(sigmas).any()
    # timestep 1 들어오면 이미지 들어온거임 -> 50번 ode sample해야 함. 
    # timestep 1000 들어오면 노이즈 들어온거임
    def sigma_to_t(sigma):
        log_sigma = torch.log(sigma)
        # get distribution
        dists = log_sigma - torch.log(sigmas)[:, None]
        # get sigmas range
        low_idx = torch.cumsum((dists >= 0), dim=0).argmax(dim=0).clamp(max=torch.log(sigmas).shape[0] - 2)
        high_idx = low_idx + 1
        low = torch.log(sigmas)[low_idx]
        high = torch.log(sigmas)[high_idx]
        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = torch.clamp(w, 0, 1)
        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        return t[0]
                
    class ODEFunc(nn.Module):
        def __init__(self, pipeline, prompt_embeds, device, verbose=False):
            super().__init__()
            self.pipeline = pipeline
            self.prompt_embeds = prompt_embeds.requires_grad_(True)
            self.nfe = 0
            self.device = device
            self.verbose = verbose
            
        # def checkpoint_forward(self, latent, timestep, encoder_hidden_states, cross_attention_kwargs):
        #     def custom_forward(*inputs):
        #         return unet(*inputs, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        #     x = checkpoint(custom_forward, x, t)
        #     return x
            
        def estimate_drift_and_divergence(self, sigma, inputs):
            with torch.enable_grad():
                timestep = sigma_to_t(sigma)
                timestep = timestep.requires_grad_()
                # epsilon = torch.randint_like(inputs[0], 2) * 2 - 1
                latent, log_p = inputs 
                latent = latent.to(self.pipeline.unet.dtype).requires_grad_(True)
                scaled_latent = latent / ((sigma ** 2 + 1) ** 0.5)        
                noise_pred = self.pipeline.unet(
                        scaled_latent,
                        int(timestep),
                        encoder_hidden_states=self.prompt_embeds, # for classifier free guidance
                        cross_attention_kwargs=None,
                        return_dict=False,
                )[0]
                drift = noise_pred
                divergence = torch.sum(
                    torch.autograd.grad(
                        torch.sum(drift * epsilon.clone()), 
                        latent,
                        create_graph=True
                    )[0] * epsilon, dim=(1,2,3)
                )
            return drift, divergence
            
        def forward(self, sigma, x):
            assert sigmas != None
            self.nfe = self.nfe + 1
            with torch.enable_grad():
                drift, divergence = self.estimate_drift_and_divergence(sigma, x)
                
                if self.verbose:
                    print("-------"*5)
                    print(f"nfe: {self.nfe}")
                    print(f"sigma: {sigma}")
                    print(f"t: {sigma_to_t(sigma)}")
                    print(f"d_ll: {divergence}")
                    print(f"ll: {x[1]}")
                # breakpoint()
                return drift, divergence
        
    ode_func = ODEFunc(pipeline, prompt_embeds, device)
    sigma_max = sigmas.max()
    sigma_min = sigmas[timestep - 1]
    assert sigmas != None
    assert not torch.isnan(sigmas).any()
    result = odeint(
        ode_func, 
        (encoded_latents.requires_grad_(), torch.zeros(encoded_latents.shape[0]).to(device).requires_grad_()),  #encoded image and the zero likelihood
        # torch.linspace(1e-4, sigma_max, steps=num_inference_steps).to(device),
        # torch.tensor([1e-4, sigma_max]).to(device),
        # torch.cat([torch.tensor([1e-4]).to(device), sigmas[:timestep - 1][::20], torch.tensor([sigma_max]).to(device)]),
        # torch.cat([torch.tensor([1e-4]).to(device), sigmas[:timestep - 1][::20]]),
        sampled_sigmas,
        method=solver,
        atol=atol,
        rtol=rtol,
    )
    
    # breakpoint()
    trajectory, delta_ll_traj = result[0], result[1] # trajectory: (50, 1, 4, 64, 64), delta_ll_traj: (50, 1)
    prior, delta_ll= trajectory[-1].unsqueeze(0), delta_ll_traj[-1]
    prior_likelihood = get_prior_likelihood(prior, sigma=sigmas.max().item()).squeeze()
    log_likelihood = delta_ll + prior_likelihood
    bpd = log_likelihood / 4 / 64 / 64 / np.log(2)
    
    # latents = torch.randn_like(encoded_latents).to(device)
    
    # result = odeint(
    #     ode_func, 
    #     (latents, torch.zeros(encoded_latents.shape[0]).to(device)),  #encoded image and the zero likelihood
    #     # torch.linspace(1e-4, sigma_max, steps=num_inference_steps).to(device),
    #     # torch.tensor([1e-4, sigma_max]).to(device),
    #     # torch.cat([torch.tensor([1e-4]).to(device), sigmas[:timestep - 1][::20], torch.tensor([sigma_max]).to(device)]),
    #     # torch.cat([torch.tensor([1e-4]).to(device), sigmas[:timestep - 1][::20]]),
    #     sampled_sigmas.flip(0),
    #     method='euler',
    #     atol=1e-3,
    #     rtol=1e-3,
    # )
    # img = image_decode(pipeline, result[0][-1])
    # plt.imsave("randn_decoding.png", img[0])
    
    # result = odeint(
    #     ode_func, 
    #     (latents * sigmas.max(), torch.zeros(encoded_latents.shape[0]).to(device)),  #encoded image and the zero likelihood
    #     # torch.linspace(1e-4, sigma_max, steps=num_inference_steps).to(device),
    #     # torch.tensor([1e-4, sigma_max]).to(device),
    #     # torch.cat([torch.tensor([1e-4]).to(device), sigmas[:timestep - 1][::20], torch.tensor([sigma_max]).to(device)]),
    #     # torch.cat([torch.tensor([1e-4]).to(device), sigmas[:timestep - 1][::20]]),
    #     sampled_sigmas.flip(0),
    #     method='euler',
    #     atol=1e-3,
    #     rtol=1e-3,
    # )
    # img = image_decode(pipeline, result[0][-1])
    # plt.imsave("randn_sigma_decoding.png", img[0])
    
    return log_likelihood, bpd, nfe, trajectory, delta_ll_traj, delta_ll, prior_likelihood, sigma_max
    
def image_decode(pipeline, latents):
    image = pipeline.vae.decode(
            latents / pipeline.vae.config.scaling_factor, return_dict=False
    )[0].detach()
    image = pipeline.image_processor.postprocess(
        image, output_type='np'
    )
    return image


## TODO sigma - timestep interpolation
## TODO refactoring
## TODO find the best sampler (euler, rk4, dopri)
## TODO sampler !!
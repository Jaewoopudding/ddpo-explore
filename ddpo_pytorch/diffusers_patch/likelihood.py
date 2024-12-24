from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
# reference: Yang Song's SDE tutorial jupyter notebook
# https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3#scrollTo=DfOkg5jBZcjF

def prior_likelihood(
    z: torch.FloatTensor, 
    sigma: float,
):
    shape = torch.tensor(z.shape)
    N = torch.prod(shape[1:])
    return -N / 2. * np.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)
    
def ode_likelihood(
    pipeline,
    x,
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
    # output_type: Optional[str] = "pil",
    # return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    shape = x.shape
    
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
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta) 
    for idx, image in enumerate(x):
        plt.imsave(f'sample_images/{idx}_original.png', image.cpu().detach().numpy().transpose(1,2,0))
        
    image_processor = VaeImageProcessor(vae_scale_factor=pipeline.vae_scale_factor)
    x = image_processor.preprocess(x, height, width)
    encoded_latents = pipeline.vae.encode( ### TODO stochasticity 제거하기
            x
    ).latent_dist.sample(generator) ###TODO we have to encode the given image and feed it to the pipeline
    
    images = image_decode(pipeline, encoded_latents)
    for idx, image in enumerate(images):
        plt.imsave(f'sample_images/{idx}_recon_original.png', image)
        
    breakpoint()
    
    class ODEFunc(nn.Module):
        def __init__(self, pipeline, prompt_embeds, device):
            super().__init__()
            self.pipeline = pipeline
            self.prompt_embeds = prompt_embeds
            self.nfe = 0
            self.device = device
            
        def estimate_score_and_divergence(self, timesteps, inputs, epsilon):
            latent_model_input, log_p = inputs 
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, timesteps) ## TODO  이부분 timestep이 아니라 sigma 접근 필요함
            latent_model_input = latent_model_input.requires_grad_(True) # 그리고 인풋을 sigma로 나누는? 상황을 연출하는데 왜그런지는 모르겠음
            # https://github.com/Mattias421/clevr_diff/blob/main/sdxl_turbo_ll.py#L64
            timesteps = timesteps.requires_grad_(True)
            print(log_p / 3 / 512 / 512 / torch.log(torch.tensor([2])).to(self.device))
            betas = self.pipeline.scheduler.betas.to(self.device)
            with torch.enable_grad():
                noise_pred = self.pipeline.unet(
                        latent_model_input,
                        timesteps, ## TODO
                        encoder_hidden_states=self.prompt_embeds,
                        cross_attention_kwargs=None,
                        return_dict=False,
                )[0]
                score = - noise_pred
                weighted_score = score * epsilon.clone()
                beta_t = betas.clone().gather(0, timesteps.to(torch.int64))
                drift = -0.5 * beta_t * latent_model_input - 0.5 * beta_t * score
                
                # print("#"*50)
                # print(f"latent_model_input: {latent_model_input.dtype}, {latent_model_input.requires_grad}")
                # print(f"timesteps: {timesteps.dtype}, {timesteps.requires_grad}")
                # print(f"prompt_embeds: {prompt_embeds.dtype}, {prompt_embeds.requires_grad}")
                # print(f"drift: {drift.grad_fn}")
                # print("#"*50)
                
                score_divergence_estimation = torch.sum(
                    epsilon * torch.autograd.grad(
                        torch.sum(weighted_score), latent_model_input
                    )[0], dim=(1,2,3)
                )
                
            
            # score_divergence_estimation = torch.sum(torch.autograd.grad(torch.sum(score * epsilon), latent_model_input)[0] * epsilon, dim=(1,2,3))
            
            
            return drift, score_divergence_estimation
            
        def forward(self, t, x):
            print("timestep:", t)
            self.nfe = self.nfe + 1
            print(self.nfe)
            epsilon = torch.randn_like(x[0])
            drift, divergence = self.estimate_score_and_divergence(t, x, epsilon)
            images = image_decode(self.pipeline, x[0])
            for idx, image in enumerate(images):
                plt.imsave(f'sample_images/{idx}_{t}.png', image)
            return drift, divergence # latent, log_p
    
    # Skilling-Hutchinson's divergence estimator
    
    log_p = torch.zeros(shape[0]).to(device)
    timesteps = pipeline.scheduler.timesteps.to(device).to(torch.float32).requires_grad_().flip(0) ## TODO: precision 16, 32
    # breakpoint()
    # timesteps = torch.tensor([0, pipeline.scheduler.config.num_train_timesteps-1]).to(device).to(torch.float32).requires_grad_()
    
    # 적응형 solver 쓸거면 (RK45) start값일아 end값만 넣어줘도 된다. 
    
    ode_func = ODEFunc(pipeline, prompt_embeds, device)
    # 왜인지 여기 debug에서는 되는데 odeint속으로 들어가면 float16, 32라던지, 안 맞는 것들이 생긴다. 
    inputs = (encoded_latents, log_p) # for debug
    ode_func(timesteps[0], inputs) # for debug
    
    print("odeint start")
    result = odeint(
        ode_func, 
        (encoded_latents, log_p), 
        timesteps, 
        method='euler',
        atol=1e-3,
        rtol=1e-3,
    )
    
    latent_prior, log_likelihood_integration = result[0][-1], result[1][-1]
    
    log_likelihood = log_likelihood_integration + prior_likelihood(latent_prior, sigma=1) ##TODO find the exact sigma.
    bpd = log_likelihood / 3 / 512 / 512 / np.log(2) + 8.
    breakpoint()
    return log_likelihood, bpd, ode_func.nfe
    
def image_decode(pipeline, latents):
    image = pipeline.vae.decode(
            latents / pipeline.vae.config.scaling_factor, return_dict=False
    )[0].detach()
    image = pipeline.image_processor.postprocess(
        image, output_type='np'
    )
    return image
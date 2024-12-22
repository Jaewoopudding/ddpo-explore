from typing import Any, Callable, Dict, List, Optional, Union

import numpy
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

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
    shape = z.shape
    N = torch.prod(shape[1:])
    return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)
    
    
# def divergence_eval(sample, timesteps, score_model, epsilon):
#     with torch.enable_grad():
#         sample.requires_grad(True)
#         score_estimation = torch.sum(score_model(sample, timesteps) * epsilon)
#         gradient_score_estimation = torch.autograd.grad(score_estimation, sample)[0] 
#     return torch.sum(gradient_score_estimation * epsilon, dim=(1,2,3))

class DiffusionProbabilisticODE(StableDiffusionPipeline):
    def forward(self, t, x):
        return super().forward(x)
    
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
    """Compute the log likelihood of the given sample with probability flow ODE

    Args:
        x (_type_): _description_
        score_model (_type_): _description_
        marginal_prob_std (_type_): _description_
        diffusion_coeff (_type_): _description_
        batch_size (_type_): _description_
        device (_type_): _description_
        eps (_type_): _description_
    """
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
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
        
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
    
    encoded_latents = pipeline.vae.encode( ### TODO stochasticity 제거하기
            x, return_dict=False
    )[0].sample() ###TODO we have to encode the given image and feed it to the pipeline
    
    # def estimate_score_and_divergence(timesteps, inputs, epsilon):
    #     latent_model_input, prompt_embeds = inputs 
    #     latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timesteps)
    #     latent_model_input.requires_grad(True) # for differentiation
    #     noise_pred = pipeline.unet(
    #             latent_model_input,
    #             timesteps,
    #             encoder_hidden_states=prompt_embeds,
    #             cross_attention_kwargs=cross_attention_kwargs,
    #             return_dict=False,
    #     )[0].sample()
        
        # if do_classifier_free_guidance:
        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (
        #         noise_pred_text - noise_pred_uncond
        #     )
        
        # if do_classifier_free_guidance and guidance_rescale > 0.0:
        #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        #     noise_pred = rescale_noise_cfg(
        #         noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
        #     )
            
        # score = - noise_pred
        # beta_t = pipeline.scheduler.betas.gather(0, timesteps)
        # drift = -0.5 * beta_t * latent_model_input - 0.5 * beta_t * score
        # score_divergence_estimation = torch.sum(
        #     torch.autograd.grad(score, latent_model_input)[0] * epsilon, dim=(1,2,3)
        # )
        # return drift, score_divergence_estimation # for returning drift and score function
    

    # def ode_func(t, x):
    #     """_summary_

    #     Args:
    #         t (_type_): _description_
    #         x (_type_): It should be a list of latent_model_input, t, prompt_embeds, cross_attention_kwargs

    #     Returns:
    #         _type_: _description_
    #     """
    #     epsilon = torch.randn_like(x[0]) # input latent dimension
    #     drift, divergence = estimate_score_and_divergence(t, x, epsilon)
    #     return drift, divergence
    
    class ODEFunc(nn.Module):
        def __init__(self, pipeline, device):
            super().__init__()
            self.pipeline = pipeline
            self.nfe = 0
            self.device = device
            
        def estimate_score_and_divergence(self, timesteps, inputs, epsilon):
            latent_model_input, prompt_embeds, _ = inputs
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, timesteps)
            # timesteps = timesteps.float().requires_grad_(True) ##TODO 제거가능한지 살펴보기
            # latent_model_input = latent_model_input.float().requires_grad_(True) # for differentiation
            # prompt_embeds = prompt_embeds.float().requires_grad_(True)
            latent_model_input = latent_model_input.requires_grad_(True)
            noise_pred = self.pipeline.unet(
                    latent_model_input,
                    timesteps, ## TODO
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
            )[0]
            # self.pipeline.unet(latent_model_input, timesteps, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=None, return_dict=False)
            # if do_classifier_free_guidance:
            #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #     noise_pred = noise_pred_uncond + guidance_scale * (
            #         noise_pred_text - noise_pred_uncond
            #     )
            
            # if do_classifier_free_guidance and guidance_rescale > 0.0:
            #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            #     noise_pred = rescale_noise_cfg(
            #         noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
            #     )
            
            score = - noise_pred
            betas = self.pipeline.scheduler.betas.to(self.device)
            beta_t = betas.gather(0, timesteps.to(torch.int64))
            drift = -0.5 * beta_t * latent_model_input - 0.5 * beta_t * score
            
            print("#"*50)
            print(f"latent_model_input: {latent_model_input.dtype}, {latent_model_input.requires_grad}")
            print(f"timesteps: {timesteps.dtype}, {timesteps.requires_grad}")
            print(f"prompt_embeds: {prompt_embeds.dtype}, {prompt_embeds.requires_grad}")
            print(f"drift: {drift.grad_fn}")
            print("#"*50)
            
            breakpoint()
            score_divergence_estimation = torch.sum(
                torch.autograd.grad(
                    torch.sum(score * epsilon), latent_model_input
                )[0] * epsilon, dim=(1,2,3)
            )
            
            
            # score_divergence_estimation = torch.sum(torch.autograd.grad(torch.sum(score * epsilon), latent_model_input)[0] * epsilon, dim=(1,2,3))
            
            
            return drift, score_divergence_estimation
            
        def forward(self, t, x):
            self.nfe = self.nfe + 1
            epsilon = torch.randn_like(x[0])
            drift, divergence = self.estimate_score_and_divergence(t, x, epsilon)
            return drift, torch.zeros_like(x[1]), divergence # latent, prompt, log_p
    
    # Skilling-Hutchinson's divergence estimator
    
    log_p = torch.zeros(shape[0]).to(device)
    timesteps = pipeline.scheduler.timesteps.to(device).float() ## TODO: precision 16, 32
    ode_func = ODEFunc(pipeline, device)
    
    # 왜인지 여기 debug에서는 되는데 odeint속으로 들어가면 float16, 32라던지, 안 맞는 것들이 생긴다. 
    inputs = (encoded_latents, prompt_embeds, log_p) # for debug
    ode_func(timesteps[0], inputs) # for debug
    
    
    result = odeint(
        ode_func, 
        (encoded_latents, prompt_embeds, log_p), 
        timesteps, 
        method='dopri5'
    )
    
    breakpoint()
    
    # 두번째 인자로 들어가는 tuple이 지속적으로 변하는 것으로 추정된다. 
    
    log_likelihood = log_likelihood_integration + prior_likelihood()
    
    return log_likelihood, bpd
    
    # def divergence_eval(inputs, timesteps, score_model, epsilon):
    #     score = score_eval_wrapper
    #     with torch.enable_grad():
    #         sample.requires_grad(True)
    #         score_estimation = torch.sum(score_model(sample, timesteps) * epsilon)
    #         gradient_score_estimation = torch.autograd.grad(score_estimation, sample)[0] 
    #     return torch.sum(gradient_score_estimation * epsilon, dim=(1,2,3))
    # def score_eval_wrapper(sample, timesteps):
    #     if isinstance(sample, np.ndarray):
    #         sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    #     if isinstance(timesteps, np.ndarray):
    #         timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    #     return score_model(sample, timesteps)
    # def divergence_eval_wrapper(sample, timesteps):
    #     if isinstance(sample, np.ndarray):
    #         sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    #     if isinstance(timesteps, np.ndarray):
    #         timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    #     return divergence_eval(sample, timesteps)
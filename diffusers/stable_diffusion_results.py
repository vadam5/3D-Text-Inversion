from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1-base"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_robot_3")
pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_lego_harry_potter_explicit_prompts_2")
pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_octopus_2")
pipe = pipe.to("cuda")

# prompts = [
#     "a <war-hammer-robot>",
#     "a <war-hammer-robot> front view",
#     "a <war-hammer-robot> viewed from the front",
#     "a <war-hammer-robot> overhead view",
#     "a <war-hammer-robot> eating a burger",
#     "a <war-hammer-robot> cooking a pot of spaghetti",
#     "a <war-hammer-robot> ballet dancing",
    
# ]

# prompts = [
#     "a <lego_harry_potter>",
#     "a <lego_harry_potter> front view",
#     "a <lego_harry_potter> viewed from the front",
#     "a <lego_harry_potter> overhead view",
#     "a <lego_harry_potter> eating a burger",
#     "a <lego_harry_potter> cooking a pot of spaghetti",
#     "a <lego_harry_potter> ballet dancing",
# ]

# prompts = [
#     "a <crochet-octopus>",
#     "a <crochet-octopus> front view",
#     "a <crochet-octopus> viewed from the front",
#     "a <crochet-octopus> overhead view",
#     "a <crochet-octopus> eating a burger",
#     "a <crochet-octopus> cooking a pot of spaghetti",
#     "a <crochet-octopus> ballet dancing",
# ]

prompts = ["a <crochet-octopus> viewed from the front",
           "a <lego_harry_potter> viewed from the front",
           "a <war-hammer-robot> viewed from the front",
           "a <crochet-octopus> eating a burger",
           "a <lego_harry_potter> eating a burger",
           "a <war-hammer-robot> eating a burger",
           "a <crochet-octopus> cooking a pot of spaghetti",
           "a <lego_harry_potter> cooking a pot of spaghetti",
           "a <war-hammer-robot> cooking a pot of spaghetti",
           "a <crochet-octopus> ballet dancing",
           "a <lego_harry_potter> ballet dancing",
           "a <war-hammer-robot> ballet dancing",
           "two <crochet-octopus>s cooking a pot of spaghetti together",
           "two <lego_harry_potter>s cooking a pot of spaghetti together",
           "two <war-hammer-robot>s cooking a pot of spaghetti together",
           "two <crochet-octopus>s ballet dancing together",
           "two <lego_harry_potter>s ballet dancing together",
           "two <war-hammer-robot>s ballet dancing together",
           "a <crochet-octopus> and a <lego_harry_potter> cooking a pot of spaghetti together",
           "a <lego_harry_potter> and a <crochet-octopus> cooking a pot of spaghetti together",
           "a <war-hammer-robot> and a <lego_harry_potter> cooking a pot of spaghetti together",
           "a <crochet-octopus> and a <war-hammer-robot> cooking a pot of spaghetti together",
           "a <war-hammer-robot> and a <crochet-octopus> cooking a pot of spaghetti together",
           "a <crochet-octopus> sitting on top of a basket of colorful macaroons",
           "a <lego_harry_potter> sitting on top of a basket of colorful macaroons",
           "a <war-hammer-robot> sitting on top of a basket of colorful macaroons",
           ]

for prompt in prompts:
    image = pipe(prompt, num_inference_steps=50).images[0]  
    
    save_name = prompt.replace(" ", "_")
    image.save(f"final_2d_results/{save_name}_stable_diffusion_2_1_base.png")

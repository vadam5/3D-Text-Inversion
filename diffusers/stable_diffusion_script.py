from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1-base"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_robot_2")
# pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_lego_harry_potter")
# pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_octopus")
pipe = pipe.to("cuda")

# prompts = [
#     "a <war-hammer-robot> facing backward"
# ]

# prompts = [
#            "a <war-hammer-robot>",
#            "a <war-hammer-robot> eating a burger",
#            "a <war-hammer-robot> cooking a pot of spaghetti",
#            "a <war-hammer-robot> ballet dancing",
#            "two <war-hammer-robot>s cooking a pot of spaghetti together",
#            "two <war-hammer-robot>s ballet dancing together",
#            "a <war-hammer-robot> sitting on top of a basket of colorful macaroons",
#            ]

prompts = ["a little orange crochet octopus",
           "a lego harry potter",
           "a war hammer robot",
           "a little orange crochet octopus eating a burger",
           "a lego harry potter eating a burger",
           "a war hammer robot eating a burger",
           "a little orange crochet octopus cooking a pot of spaghetti",
           "a lego harry potter cooking a pot of spaghetti",
           "a war hammer robot cooking a pot of spaghetti",
           "a little orange crochet octopus ballet dancing",
           "a lego harry potter ballet dancing",
           "a war hammer robot ballet dancing",
           "two little orange crochet octopuses cooking a pot of spaghetti together",
           "two lego harry potters cooking a pot of spaghetti together",
           "two war hammer robots cooking a pot of spaghetti together",
           "two little orange crochet octopuses ballet dancing together",
           "two lego harry potters ballet dancing together",
           "two war hammer robots ballet dancing together",
           "a little orange crochet octopus and a lego harry potter cooking a pot of spaghetti together",
           "a lego harry potter and a little orange crochet octopus cooking a pot of spaghetti together",
           "a war hammer robot and a lego harry potter cooking a pot of spaghetti together",
           "a little orange crochet octopus and a war hammer robot cooking a pot of spaghetti together",
           "a war hammer robot and a little orange crochet octopus cooking a pot of spaghetti together",
           "a little orange crochet octopus sitting on top of a basket of colorful macaroons",
           "a lego harry potter sitting on top of a basket of colorful macaroons",
           "a war hammer robot sitting on top of a basket of colorful macaroons",

           ]

for prompt in prompts:
    image = pipe(prompt, num_inference_steps=50).images[0]  
    
    save_name = prompt.replace(" ", "_")
    image.save(f"stable_diffusion_baselines/{save_name}_stable_diffusion_2_1_base.png")

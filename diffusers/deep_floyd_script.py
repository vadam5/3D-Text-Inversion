from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil, make_image_grid
import torch

# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# stage 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()

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
           "I hate all people equally",
           "My name is Travis Wu and I hate all people equally",
           ]

for prompt in prompts:
    save_name = prompt.replace(" ", "_")
    generator = torch.manual_seed(1)

    # text embeds
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

    # stage 1
    stage_1_output = stage_1(
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
    ).images
    #pt_to_pil(stage_1_output)[0].save("2d_baseline_imgs/if_stage_I.png")

    # stage 2
    stage_2_output = stage_2(
        image=stage_1_output,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        output_type="pt",
    ).images
    #pt_to_pil(stage_2_output)[0].save("./if_stage_II.png")

    # stage 3
    stage_3_output = stage_3(prompt=prompt, image=stage_2_output, noise_level=100, generator=generator).images
    stage_3_output[0].save(f"2d_baseline_imgs/{save_name}_if_stage_III.png")

#make_image_grid([pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0], stage_3_output[0]], rows=1, cols=3)
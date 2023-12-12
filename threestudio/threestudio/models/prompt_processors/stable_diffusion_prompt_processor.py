import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-prompt-processor")
class StableDiffusionPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model_id = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_robot_3")
        pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_lego_harry_potter_explicit_prompts_2")
        pipe.load_textual_inversion("/home/ubuntu/3d_text_inversion/diffusers/examples/textual_inversion/textual_inversion_octopus_2")
        #pipe = pipe.to("cuda")

        # tokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path, subfolder="tokenizer"
        # )
        # text_encoder = CLIPTextModel.from_pretrained(
        #     pretrained_model_name_or_path,
        #     subfolder="text_encoder",
        #     device_map="auto",
        # )

        with torch.no_grad():
            tokens = pipe.tokenizer(
                prompts,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            )

            text_embeddings = pipe.text_encoder(tokens.input_ids.to(pipe.text_encoder.device))[0]
            # text_embeddings = pipe.encode_prompt(
            #     prompts, 
            #     device=pipe.device, 
            #     num_images_per_prompt=1, 
            #     do_classifier_free_guidance=False,
            # )
            
        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )
            
        del pipe
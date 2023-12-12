import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from PIL import Image

# Load the CLIP model
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)

preprocess = CLIPImageProcessor.from_pretrained(model_ID)

# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)

    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")

    # Return the preprocessed image
    return image

# Load the two images and preprocess them for CLIP
image_a = load_and_preprocess_image('/home/ubuntu/3d_text_inversion/gt_images/robot/IMG_6995.jpg')["pixel_values"]
image_b = load_and_preprocess_image('/home/ubuntu/3d_text_inversion/all_result_imgs/stable_diffusion_2d_results/final_2d_results/a_<war-hammer-robot>_eating_a_burger_stable_diffusion_2_1_base.png')["pixel_values"]

# Calculate the embeddings for the images using the CLIP model
with torch.no_grad():
    embedding_a = model.get_image_features(image_a)
    embedding_b = model.get_image_features(image_b)

# Calculate the cosine similarity between the embeddings
similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)

# Print the similarity score
print('Similarity score:', similarity_score.item())

import torch
import transformers
import torchvision.transforms as transforms
import argparse
from PIL import Image
from torch.nn.functional import cosine_similarity
import math

def clip(args):

    logit_scale = 2.6592
    logit_scale = torch.tensor(logit_scale).exp().square()
    
    vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)) # for denormalizing images


    label = args.label
    image = Image.open(f'{args.image}.jpeg').convert('RGB')
    processed_image = test_transform(image).cuda().half().unsqueeze(0)

    # projection_layer = torch.nn.Linear(768, 1024, bias=False).cuda().half()
    projection_layer = torch.nn.Linear(768, 1024, bias=True).cuda().half()

    vision_model.eval()
    text_model.eval()

    with torch.no_grad():
        text_labels = tokenizer(label, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_embeds = text_model(text_labels).text_embeds

        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        animal_representation = text_embeds

        print(f'animal_representation shape: {animal_representation.shape}')

        animal_representation = projection_layer(animal_representation)

        print(f'animal_representation shape after projection: {animal_representation.shape}')


        vision_output = vision_model(processed_image)
        last_hidden_state = vision_output.last_hidden_state

        # print(f'image embeds shape: {vision_output.image_embeds.shape}')

        animal_representation = animal_representation
        all_patches = last_hidden_state[:, 1:257, :].squeeze(0)

        # print(f'all pathces shape: {all_patches.shape}')

        similarities = cosine_similarity(animal_representation, all_patches)

        similarities = similarities * logit_scale

        similarities = similarities.softmax(dim=0)
        
        top_similarities, top_indices = torch.topk(similarities, k=64)

        mask = torch.zeros(256, dtype=torch.bool).cuda()

        mask[top_indices] = True
        
        # Reshaping the mask to match the image size (16x16 patches)
        mask_2d = mask.view(16, 16)
        
        # Upscale the mask to match the image size
        upscale = transforms.Resize((224, 224), interpolation=Image.NEAREST)
        mask_image = Image.fromarray(mask_2d.cpu().float().numpy() * 255).convert("L")
        mask_image = upscale(mask_image)
        mask_tensor = transforms.ToTensor()(mask_image).cuda().half()

        # Ensure mask_tensor is of shape [224, 224]
        mask_tensor = mask_tensor.squeeze()

        # Reshape mask tensor dimensions to [1, 1, 224, 224]
        mask_tensor_reshaped = mask_tensor.unsqueeze(0).unsqueeze(1)

        # Expand mask tensor dimensions to [1, 3, 224, 224] to match processed_image dimensions
        mask_tensor_expanded = mask_tensor_reshaped.expand(-1, 3, -1, -1)

        # Compute the mean of the relevant patches using the expanded mask
        relevant_pixels = processed_image * mask_tensor_expanded
        mean_relevant_patch = relevant_pixels.sum(dim=[2, 3]) / mask_tensor_expanded.sum(dim=[2, 3])

        # Replace irrelevant patches with the mean of the relevant patches
        masked_image_tensor = relevant_pixels + (1 - mask_tensor_expanded) * mean_relevant_patch.unsqueeze(-1).unsqueeze(-1)

        masked_image_tensor = denormalize(masked_image_tensor)

        # Convert tensor to image for saving
        masked_image = transforms.ToPILImage()(masked_image_tensor.squeeze(0).cpu().float())
        masked_image.save(f"masked_{args.image}.jpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="elephant", required=False)
    parser.add_argument("--label", type=str, default="animal", required=False)
    args = parser.parse_args()

    clip(args)

from tqdm import tqdm
import argparse
import torch
import os
import torchvision.transforms as T
import numpy as np
import json
import transformers
from PIL import Image

def get_different_class(c_true, classes):
    classes_kept = [c for c in classes if c != c_true]
    chosen_class = np.random.choice(classes_kept)
    return classes.index(chosen_class) 

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def print_clip_top_probs(logits_per_image, classes):
    """
    Print the top 5 class probabilities for a given image based on its logits.
    :param logits_per_image: Tensor of logits per image generated by the vision model.
    :param classes: List of class names corresponding to each logit in the tensor.
    """

    probs = logits_per_image.softmax(dim=1)
    top_probs_scaled, top_idxs_scaled = probs[0].topk(5)

    if classes != None:
        for i in range(top_probs_scaled.shape[0]):
            print(f"{classes[top_idxs_scaled[i]]}: {top_probs_scaled[i].item() * 100:.2f}%")   


def clip_model_fn(x, text_embeds, vision_model, classes=None):
    """
    Compute CLIP similarity scores between an image and a set of text embeddings.
    :param x: Image tensor.
    :param text_embeddings: Precomputed text embeddings for all labels.
    :return: Similarity scores between the image and all text labels.
    """

    logit_scale = 2.6592
    logit_scale = torch.tensor(logit_scale).exp()

    image_embeds = vision_model(x).image_embeds.half()
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale

    print_clip_top_probs(logits_per_image, classes)

    return logits_per_image

def generate_adversary(image, text_embeddings, vision_model, n_classes):

    adv_image = carlini_wagner_l2(
        model_fn=lambda x: clip_model_fn(x, text_embeddings, vision_model, classes=None), 
        x=image,
        n_classes=n_classes,
        lr=5e-3,
        confidence=1,
        initial_const=1e-2,
        binary_search_steps=5,
        max_iterations=100
    )
    return adv_image


# Generate many adversarial examples using PGD attack
def generate_adversarials_pgd(args):

    f = open(args.label_list, 'r')
    label_names = list(json.load(f).keys())
    f.close()

    label_all = []

    for v in label_names:
        label_all.append("a photo of %s"%v)

    f = open(args.image_list, 'r')
    image_list = f.readlines()
    f.close()

    vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    test_transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        # T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    denormalize = T.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)) # for denormalizing images
     
    vision_model.eval()
    text_model.eval()

    image_list = image_list[args.set*args.set_num:(args.set+1)*args.set_num]

    with torch.no_grad():
        text_labels = tokenizer(label_all, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = text_model(text_labels).text_embeds
        text_label_embeds = text_label_embeds / text_label_embeds.norm(p=2, dim=-1, keepdim=True)

    for i, data in tqdm( enumerate(image_list)):
        image_name, label_name = data.split('|')
        print(f'Looking at image: {image_name}')
        image = Image.open(image_name).convert('RGB')
        image = test_transform(image).cuda().half().unsqueeze(0)
        # image = test_transform(image).cuda().unsqueeze(0)

        ## Text label embedding
        label_name = label_name.split('\n')[0]
        label_name = "a photo of " + label_name

        # Similarities pre-attack
        print('\nProbabilities pre-attack:')
        logits_per_image = clip_model_fn(image, text_label_embeds, vision_model, label_all)

        # Generate adversarial
        adv_image = generate_adversary(image=image,
                                       text_embeddings=text_label_embeds,
                                        vision_model=vision_model,
                                        n_classes = len(label_all))
        
        print("Probabilities post-attack")
        # Similarities post-attackf
        denormalized_tensor = denormalize(adv_image)

        logits_per_image = clip_model_fn(denormalized_tensor, text_label_embeds, vision_model, label_all)

        save_image = denormalized_tensor.squeeze(0)
        save_image = T.ToPILImage()(save_image)

        base_name = os.path.basename(image_name)

        if args.save_image: 
            os.makedirs(f"/home/crcvreu.student2/coco/{args.attack_type}_rgb", exist_ok=True)
            save_image.save(f'/home/crcvreu.student2/coco/{args.attack_type}_rgb/{base_name}')

        tmp = image_name.split(os.sep)

        os.makedirs(f"/home/crcvreu.student2/coco/carlini_wagner", exist_ok=True)
        tmp[tmp.index('trainval2014')] = 'carlini_wagner'
        
        save_name = os.path.splitext(os.sep + os.path.join(*tmp))[0] + ".pt"
        # torch.save(adv_image, save_name)
        torch.save(denormalized_tensor, save_name)

INF = float("inf")

def carlini_wagner_l2(
    model_fn,
    x,
    n_classes,
    y=None,
    targeted=False,
    lr=5e-3,
    confidence=0,
    clip_min=0,
    clip_max=1,
    initial_const=1e-2,
    binary_search_steps=5,
    max_iterations=1000,
):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644

    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.

    :param model_fn: a callable that takes an input tensor and returns
              the model logits. The logits should be a tensor of shape
              (n_examples, n_classes).
    :param x: input tensor of shape (n_examples, ...), where ... can
              be any arbitrary dimension that is compatible with
              model_fn.
    :param n_classes: the number of classes.
    :param y: (optional) Tensor with true labels. If targeted is true,
              then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when
              crafting adversarial samples. Otherwise, model predictions
              are used as labels to avoid the "label leaking" effect
              (explained in this paper:
              https://arxiv.org/abs/1611.01236). If provide y, it
              should be a 1D tensor of shape (n_examples, ).
              Default is None.
    :param targeted: (optional) bool. Is the attack targeted or
              untargeted? Untargeted, the default, will try to make the
              label incorrect. Targeted will instead try to move in the
              direction of being more like y.
    :param lr: (optional) float. The learning rate for the attack
              algorithm. Default is 5e-3.
    :param confidence: (optional) float. Confidence of adversarial
              examples: higher produces examples with larger l2
              distortion, but more strongly classified as adversarial.
              Default is 0.
    :param clip_min: (optional) float. Minimum float value for
              adversarial example components. Default is 0.
    :param clip_max: (optional) float. Maximum float value for
              adversarial example components. Default is 1.
    :param initial_const: The initial tradeoff-constant to use to tune the
              relative importance of size of the perturbation and
              confidence of classification. If binary_search_steps is
              large, the initial constant is not important. A smaller
              value of this constant gives lower distortion results.
              Default is 1e-2.
    :param binary_search_steps: (optional) int. The number of times we
              perform binary search to find the optimal tradeoff-constant
              between norm of the perturbation and confidence of the
              classification. Default is 5.
    :param max_iterations: (optional) int. The maximum number of
              iterations. Setting this to a larger value will produce
              lower distortion results. Using only a few iterations
              requires a larger learning rate, and will produce larger
              distortion results. Default is 1000.
    """

    def compare(pred, label, is_logits=False):
        """
        A helper function to compare prediction against a label.
        Returns true if the attack is considered successful.

        :param pred: can be either a 1D tensor of logits or a predicted
                class (int).
        :param label: int. A label to compare against.
        :param is_logits: (optional) bool. If True, treat pred as an
                array of logits. Default is False.
        """

        # Convert logits to predicted class if necessary
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -confidence if targeted else confidence
            pred = torch.argmax(pred_copy)

        return pred == label if targeted else pred != label

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        pred = model_fn(x)
        y = torch.argmax(pred, 1)

    # Initialize some values needed for binary search on const
    lower_bound = [0.0] * len(x)
    upper_bound = [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const

    o_bestl2 = [INF] * len(x)
    o_bestscore = [-1.0] * len(x)
    x = torch.clamp(x, clip_min, clip_max)
    ox = x.clone().detach()  # save the original x
    o_bestattack = x.clone().detach()

    # Map images into the tanh-space
    x = (x - clip_min) / (clip_max - clip_min)
    x = torch.clamp(x, 0, 1)
    x = x * 2 - 1
    x = torch.arctanh(x * 0.999999)

    # Prepare some variables
    modifier = torch.zeros_like(x, requires_grad=True)
    y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float)

    # Define loss functions and optimizer
    f_fn = lambda real, other, targeted: torch.max(
        ((other - real) if targeted else (real - other)) + confidence,
        torch.tensor(0.0).to(real.device),
    )
    l2dist_fn = lambda x, y: torch.pow(x - y, 2).sum(list(range(len(x.size())))[1:])
    optimizer = torch.optim.Adam([modifier], lr=lr)

    # Outer loop performing binary search on const
    for outer_step in range(binary_search_steps):
        # Initialize some values needed for the inner loop
        bestl2 = [INF] * len(x)
        bestscore = [-1.0] * len(x)

        # Inner loop performing attack iterations
        for i in range(max_iterations):
            # One attack step
            new_x = (torch.tanh(modifier + x) + 1) / 2
            new_x = new_x * (clip_max - clip_min) + clip_min
            logits = model_fn(new_x)

            real = torch.sum(y_onehot * logits, 1)
            other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)

            optimizer.zero_grad()
            f = f_fn(real, other, targeted)
            l2 = l2dist_fn(new_x, ox)
            loss = (const * f + l2).sum()
            loss.backward()
            optimizer.step()

            # Update best results
            for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                y_n = y[n]
                succeeded = compare(logits_n, y_n, is_logits=True)
                if l2_n < o_bestl2[n] and succeeded:
                    pred_n = torch.argmax(logits_n)
                    o_bestl2[n] = l2_n
                    o_bestscore[n] = pred_n
                    o_bestattack[n] = new_x_n
                    # l2_n < o_bestl2[n] implies l2_n < bestl2[n] so we modify inner loop variables too
                    bestl2[n] = l2_n
                    bestscore[n] = pred_n
                elif l2_n < bestl2[n] and succeeded:
                    bestl2[n] = l2_n
                    bestscore[n] = torch.argmax(logits_n)

        # Binary search step
        for n in range(len(x)):
            y_n = y[n]

            if compare(bestscore[n], y_n) and bestscore[n] != -1:
                # Success, divide const by two
                upper_bound[n] = min(upper_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
            else:
                # Failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lower_bound[n] = max(lower_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
                else:
                    const[n] *= 10

    return o_bestattack.detach()

# Taken from cleverhans library
def zero_out_clipped_grads(grad, x, clip_min, clip_max):
    """
    Helper function to erase entries in the gradient where the update would be
    clipped.
    :param grad: The gradient
    :param x: The current input
    :param clip_min: Minimum input component value
    :param clip_max: Maximum input component value
    """
    signed_grad = torch.sign(grad)

    # Find input components that lie at the boundary of the input range, and
    # where the gradient points in the wrong direction.
    clip_low = torch.le(x, clip_min) & torch.lt(signed_grad, 0)
    clip_high = torch.ge(x, clip_max) & torch.gt(signed_grad, 0)
    clip = clip_low | clip_high
    grad = torch.where(clip, torch.zeros_like(grad), grad)

    return grad

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_name", type=str, default="zebra.jpeg")
    parser.add_argument("--image_list", type=str, default="./coco_test.txt")
    parser.add_argument("--label_list", type=str, default="./coco_label.json")
    parser.add_argument("--attack_type", type=str, default="carlini_wagner", required=False)
    parser.add_argument("--set", type=int, required=True)
    parser.add_argument("--set_num", type=int, required=True)
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False')

    args = parser.parse_args()

    generate_adversarials_pgd(args)

    # python carlini_wagner.py --set 0 --set_num 1 --save_image True
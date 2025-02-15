"""
Prepared by:    Alejandro Aparcedo
Date:           9/22/2023
Description:    Generates adversarial examples of COCO validation set using functions taken from cleverhans library.
                The attack in this case is Projected Gradient Descent.
                **USE THIS FILE FOR ABLATION**
"""

from tqdm import tqdm
import transformers
import json
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
from llava.utils import disable_torch_init
from llava.model import *
import torchvision.transforms as transforms
import sys

mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1,3,1,1))
std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1,3,1,1))

denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)) # for denormalizing images

criterion = torch.nn.CrossEntropyLoss()

test_true_label = torch.tensor('a photo of a motorcycle').unsqueeze(0).cuda()
test_target_label = torch.tensor('a photo of a toilet').unsqueeze(0).cuda()

def get_different_class(c_true, classes):
    classes_kept = [c for c in classes if c != c_true]
    new_class_idx = np.random.choice(classes_kept)
    new_label = classes.index(new_class_idx) 
    new_label = torch.tensor(new_label).unsqueeze(0).cuda()
    return new_label

def print_clip_top_probs(logits_per_image, classes):
    """
    Print the top 5 class probabilities for a given image based on its logits.
    :param logits_per_image: Tensor of logits per image generated by the vision model.
    :param classes: List of class names corresponding to each logit in the tensor.
    """

    logit_scale = 2.6592
    logit_scale = torch.tensor(logit_scale).exp()

    logits_per_image *= logit_scale

    probs = logits_per_image.softmax(dim=1)
    top_probs_scaled, top_idxs_scaled = probs[0].topk(5)

    if classes != None:
        for i in range(top_probs_scaled.shape[0]):
            print(f"{classes[top_idxs_scaled[i]]}: {top_probs_scaled[i].item() * 100:.2f}%")   


def clip_model_fn(x, text_embeddings, vision_model, classes=None):
    """
    Compute CLIP similarity scores between an image and a set of text embeddings.
    :param x: Image tensor.
    :param text_embeddings: Precomputed text embeddings for all labels.
    :return: Similarity scores between the image and all text labels.
    """

    vision_model = vision_model
    image_embeds = vision_model(x).image_embeds
    logits_per_image = torch.matmul(image_embeds, text_embeddings.t())
    # logits_per_image = torch.matmul(image_embeds, text_embeddings.t()) * logit_scale

    print_clip_top_probs(logits_per_image, classes)

    return logits_per_image

# Generate PGD adversary for image
def generate_adversary(text_embeddings, image, vision_model, target_label):
    """
    Generate an adversarial example for the provided image using the PGD attack.
    """
    # Carry out the PGD attack
    adv_image = projected_gradient_descent(
        model_fn=lambda x: clip_model_fn(x, text_embeddings, vision_model, classes=None), 
        x=image,
        eps=0.5,   # You can adjust epsilon as needed
        eps_iter=0.01,   # You can adjust eps_iter as needed
        nb_iter=30,   # Number of PGD iterations
        norm=np.inf,   # Infinity norm
        y=target_label,
        targeted=True
    )

    return adv_image

# Generate many adversarial examples using PGD attack
def generate_adversarials_pgd(args):

    label_all = []

    f = open(args.image_list, 'r')
    image_list = f.readlines()
    f.close()

    if args.descriptors:
        descriptor_list = f'descriptors/descriptors_{args.dataset}.json'
        f = open(args.descriptor_list, 'r')
        descriptor_list = list(json.load(f).keys())
        f.close()
        label_all = descriptor_list
    else: 
        f = open(args.label_list, 'r')
        label_names = list(json.load(f).keys())
        f.close()
        for v in label_names:
            label_all.append("a photo of %s"%v)
    

    disable_torch_init()

    vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
     
    vision_model.eval()
    text_model.eval()

    image_list = image_list[args.set*args.set_num:(args.set+1)*args.set_num]

    with torch.no_grad():
        text_labels = tokenizer(label_all, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = text_model(text_labels).text_embeds
        text_label_embeds = text_label_embeds / text_label_embeds.norm(p=2, dim=-1, keepdim=True)

    for i, data in tqdm( enumerate(image_list)):
        image_name, label_name = data.split('|')
        base_name = os.path.basename(image_name)
        image = Image.open(image_name).convert('RGB')
        image = test_transform(image).cuda().half().unsqueeze(0)

        ## Text label embedding
        label_name = label_name.split('\n')[0]
        label_name = "a photo of " + label_name

        # Uncomment to look at CLIP similaritiy probabilities PRE-attack
        print('\nProbabilities pre-attack:')
        logits_per_image = clip_model_fn(image, text_label_embeds, vision_model, label_all)

        # Get a new label for our targeted attack (used in pgd function call)
        target_label = get_different_class(label_name, label_all)
        print(f'old label: {label_name}, target_label: {target_label}')

        # Generate adversarials
        adv_image = generate_adversary( text_embeddings=text_label_embeds,
                                        image=image,
                                        vision_model=vision_model,
                                        target_label=target_label
                                        )
        

        denormalized_tensor = denormalize(adv_image)

        # Uncomment to look at CLIP similaritiy probabilities POST-attack
        print("Probabilities post-attack")
        logits_per_image = clip_model_fn(denormalized_tensor, text_label_embeds, vision_model, label_all)

        if args.save_image: 
            save_image = denormalized_tensor.squeeze(0)
            save_image = T.ToPILImage()(save_image)
            os.makedirs(f"/home/crcvreu.student2/coco/pgd_0.5_0.0.1_30_rgb", exist_ok=True)
            save_image.save(f'/home/crcvreu.student2/coco/pgd_0.5_0.0.1_30_rgb/{base_name}')
            # save_image.save(f'/home/crcvreu.student2/coco/test/{base_name}')

        tmp = image_name.split(os.sep)
        os.makedirs(f"/home/crcvreu.student2/coco/pgd_0.5_0.0.1_30", exist_ok=True) # pgd should give same results as pgd_denorm
        tmp[tmp.index('trainval2014')] = 'pgd_0.5_0.0.1_30'
        # tmp[tmp.index('trainval2014')] = 'test'
        save_name = os.path.splitext(os.sep + os.path.join(*tmp))[0] + ".pt"
        torch.save(denormalized_tensor, save_name)

        sys.exit()

"""Utils for PyTorch"""
# Taken from cleverhans library
def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.
    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta

# Taken from cleverhans library
def get_or_guess_labels(model, x, **kwargs):
    """
    Get the label to use in generating an adversarial example for x.
    The kwargs are fed directly from the kwargs of the attack.
    If 'y' is in kwargs, then assume it's an untargeted attack and
    use that as the label.
    If 'y_target' is in kwargs and is not none, then assume it's a
    targeted attack and use that as the label.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack.

    :param model: PyTorch model. Do not add a softmax gate to the output.
    :param x: Tensor, shape (N, d_1, ...).
    :param y: (optional) Tensor, shape (N).
    :param y_target: (optional) Tensor, shape (N).
    """
    if "y" in kwargs and "y_target" in kwargs:
        raise ValueError("Can not set both 'y' and 'y_target'.")
    if "y" in kwargs:
        labels = kwargs["y"]
    elif "y_target" in kwargs and kwargs["y_target"] is not None:
        labels = kwargs["y_target"]
    else:
        _, labels = torch.max(model(x), 1)
    return labels

# Taken from cleverhans library
def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float16)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(torch.float16) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float16)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation

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

"""The Fast Gradient Method attack."""
# Taken from cleverhans library
def fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float16).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x), y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    
    test_loss = True
    if test_loss:
        loss_old_label = loss_fn(model_fn(x), test_true_label) # This should go up / stay the same
        loss_new_label = loss_fn(model_fn(x), test_target_label) # This should definitely be going down
        print(f'old_label_loss: {loss_old_label}')
        print(f'loss_new_label: {loss_new_label}')


    # Define gradient of loss wrt input
    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x

"""The Projected Gradient Descent attack."""
# Taken from cleverhans library
def projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        # print(f'step: {i}')
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        # print(f'eta pre clip_eta: {eta}')
        eta = clip_eta(eta, norm, eps) # Commenting this out made the attack work. Why?
        # print(f'eta post clip_eta: {eta}')
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_name", type=str, default="zebra.jpeg")
    parser.add_argument("--image_list", type=str, default="./coco_test.txt")
    parser.add_argument("--label_list", type=str, default="./coco_label.json")
    parser.add_argument("--attack_type", type=str, default="pgd")
    parser.add_argument("--set", type=int, required=True)
    parser.add_argument("--set_num", type=int, required=True)
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False')
    parser.add_argument("--descriptors", type=boolean_string, required=False, default='False')

    args = parser.parse_args()

    generate_adversarials_pgd(args)
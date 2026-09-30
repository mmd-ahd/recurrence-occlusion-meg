"""ImageNet-1k data pipeline and evaluation helpers shared by HTRN training and evaluation."""
import io

import torch
from PIL import Image
from torchvision import transforms

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Training augmentation (includes random erasing as synthetic occlusion)
train_post_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    _NORMALIZE,
    transforms.RandomErasing(p=0.2),
])

# Clean validation images
val_clean_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])

# Validation with synthetic occlusion on every image (p=1.0), to measure robustness
val_occluded_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
    transforms.RandomErasing(p=1.0),
])


def extract_img(image_item):
    """PIL RGB image from a Parquet row, whether stored as raw bytes or as an image."""
    if isinstance(image_item, dict) and "bytes" in image_item:
        return Image.open(io.BytesIO(image_item["bytes"])).convert("RGB")
    return image_item.convert("RGB")


def _apply(transform):
    def fn(batch):
        batch["pixel_values"] = [transform(extract_img(img)) for img in batch["image"]]
        return batch
    return fn


# Batch transforms for Hugging Face ``Dataset.set_transform``
transform_train_fn = _apply(train_post_transform)
transform_val_clean_fn = _apply(val_clean_transform)
transform_val_occluded_fn = _apply(val_occluded_transform)


def collate_fn(batch):
    """Stack a batch of dataset rows into ``(pixel_values, labels)`` tensors."""
    pixel_values = torch.stack([
        item["pixel_values"].clone().detach() if isinstance(item["pixel_values"], torch.Tensor)
        else torch.tensor(item["pixel_values"])
        for item in batch
    ])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return pixel_values, labels


def compute_accuracy(output, target, topk=(1, 5)):
    """Top-k accuracy (%) of a batch of logits for each k in ``topk``."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def run_evaluation(model, dataloader, device, iterations):
    """Mean top-1 and top-5 accuracy (%) of ``model(images, T=iterations)`` over ``dataloader``."""
    model.eval()
    top1_sum, top5_sum, total = 0.0, 0.0, 0

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(imgs, T=iterations)

            top1, top5 = compute_accuracy(outputs, targets, topk=(1, 5))
            n = targets.size(0)
            top1_sum += top1 * n
            top5_sum += top5 * n
            total += n

    return top1_sum / total, top5_sum / total


def run_deterministic_occluded_evaluation(model, dataloader, device, iterations, eval_seed=715):
    """Occluded evaluation with the RNG reset first, so every epoch sees identical erasing patches."""
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)
    return run_evaluation(model, dataloader, device, iterations)

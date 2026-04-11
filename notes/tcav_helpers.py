"""Helper functions for concept activation vectors and TCAV analysis."""

from pathlib import Path
from torchvision import transforms
import PIL
import glob


def load_tensor(filename):
    """Load a single image file as a tensor"""
    img = PIL.Image.open(filename).convert("RGB")
    return transform(img)


def load_tensors(class_name, root_path="data/concepts/", transform_flag=True):
    """Load all the images belonging to a class as a tensor. This assumes that
    each class gets a subdirectory of root_path."""
    path = Path(root_path) / class_name
    filenames = glob.glob(str(path / '*.jpg'))

    tensors = []
    for filename in filenames:
        img = PIL.Image.open(filename).convert('RGB')
        tensors.append(transform(img) if transform_flag else img)
    return tensors


def transform(img):
    """Transform an image into a form that can be used for classification"""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)

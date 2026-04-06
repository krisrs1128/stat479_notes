"""Implementation details for the solar-farm mapping notebook.

Adapted from microsoft/solar-farms-mapping (MIT License).
"""
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.mask import mask as rio_mask
import shapely
from shapely import geometry
import fiona
import cv2
from skimage.measure import find_contours
from skimage.transform import resize
from skimage.draw import polygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RASTERIO_BEST_PRACTICES = dict(
    CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_MAX_RAW_BLOCK_CACHE_SIZE="200000000",
    GDAL_SWATH_SIZE="200000000",
    VSI_CURL_CACHE_SIZE="200000000",
)

SENTINEL_URL = (
    "https://researchlabwuopendata.blob.core.windows.net/"
    "sentinel-2-imagery/karnataka_change/2020/2020_merged.tif"
)

MEAN = np.array([660.5929, 812.9481, 1080.6552, 1398.3968, 1662.5913, 1899.4804,
                 2061.932, 2100.2792, 2214.9325, 2230.5973, 2443.3014, 1968.1885])
STD = np.array([137.4943, 195.3494, 241.2698, 378.7495, 383.0338, 449.3187,
                511.3159, 547.6335, 563.8937, 501.023, 624.041, 478.9655])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        return self.conv(torch.cat([skip, x], 1))


class UnetModel(nn.Module):
    def __init__(self, input_channels=12, first_layer_filters=16, net_depth=5, num_classes=2):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        in_c, out_c = input_channels, first_layer_filters
        for _ in range(net_depth):
            self.downblocks.append(ConvBlock(in_c, out_c))
            in_c, out_c = out_c, 2 * out_c

        self.middle_conv = ConvBlock(in_c, out_c)

        in_c, out_c = out_c, out_c // 2
        for _ in range(net_depth):
            self.upblocks.append(UpBlock(in_c, out_c))
            in_c, out_c = out_c, out_c // 2

        self.seg_layer = nn.Conv2d(2 * out_c, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for op in self.downblocks:
            skips.append(op(x))
            x = self.pool(skips[-1])
        x = self.middle_conv(x)
        for op in self.upblocks:
            x = op(x, skips.pop())
        return self.seg_layer(x)


def load_model(checkpoint_path):
    model = UnetModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support checkpoints saved with older naming where ConvBlock used
    # "conv_block" and included extra (unused) conv1/conv2 entries.
    state_dict = checkpoint.get("model", checkpoint)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model_keys = set(model.state_dict().keys())
    remapped = {}
    for key, value in state_dict.items():
        mapped = key.replace(".conv_block.", ".block.")
        if mapped in model_keys:
            remapped[mapped] = value

    model.load_state_dict(remapped, strict=False)
    model.eval()
    return model


def scale(x, min_val, max_val, a=0, b=255):
    y = np.clip((x - min_val) / (max_val - min_val), 0, 1)
    return ((b - a) * y + a).astype(np.uint8)


def get_all_geoms(path):
    with fiona.open(path) as f:
        return [row["geometry"] for row in f]


def get_sentinel_image(geom, buffer, url=SENTINEL_URL):
    footprint = geometry.shape(geom).buffer(0.0)
    bounding = footprint.envelope.buffer(buffer).envelope
    bounding_geom = shapely.geometry.mapping(bounding)
    with rasterio.Env(**RASTERIO_BEST_PRACTICES):
        with rasterio.open(url) as f:
            image, _ = rio_mask(f, [bounding_geom], crop=True, all_touched=True)
    return np.rollaxis(image, 0, 3)


def preprocess(image, size=256):
    img = cv2.resize(image, (size, size))
    x = (img - MEAN) / STD
    x = np.moveaxis(x, 2, 0)
    return img, torch.from_numpy(x).float().unsqueeze(0).to(device)


def predict_mask(model, x_tensor, min_area=5):
    with torch.no_grad():
        logits = model(x_tensor)
    y_hat = np.argmax(logits.cpu().numpy(), axis=1).squeeze()
    mask = np.zeros_like(y_hat, dtype=float)
    for contour in find_contours(y_hat, 0.5):
        ll, ur = np.min(contour, 0), np.max(contour, 0)
        if np.prod(ur - ll) < min_area:
            continue
        rr, cc = polygon(contour[:, 0], contour[:, 1], mask.shape)
        mask[rr, cc] = 1
    return mask


def plot_sample_prediction(image, pred):
    image = resize(image, (512, 512, 3), anti_aliasing=True)
    pred = resize(pred, (512, 512), anti_aliasing=True)
    cmap = ListedColormap(sns.color_palette(["#3498db", "#FFD700"]).as_hex())
    legend = [Line2D([0], [0], marker="o", color="w", label="Solar Farm Prediction",
                     markerfacecolor=cmap(1), markersize=15)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].imshow(np.ma.masked_where(pred == 0, pred), alpha=0.5,
                   interpolation="none", cmap=cmap, vmin=0, vmax=1)
    axes[1].axis("off")
    axes[1].legend(handles=legend, bbox_to_anchor=(1, 1))
    plt.tight_layout()

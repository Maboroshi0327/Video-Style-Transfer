import torch

import os
import csv
import argparse
import numpy as np
from PIL import Image

from vgg19 import VGG19
from network import StylizingNetwork
from utilities import toTensor255, toPil, mkdir
from eval import lpips_loss, ssim_loss, kl_loss, gram_loss, nth_order_moment, uniformity, average_entropy


MODEL_EPOCH = 10
BATCH_SIZE = 8
MODEL_PATH = f"./models/AdaAttN-image_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

IMAGE_SIZE = (512, 512)
ACTIAVTION = "softmax"

CONTENT_STYLE_PAIR = [
    ("./contents/Chair.jpg", "./styles/Brushstrokes.jpg"),
    ("./contents/Brad-Pitt.jpg", "./styles/Sketch.jpg"),
    ("./contents/Bird.jpg", "./styles/Tableau.jpg"),
]

opt = argparse.Namespace(
    path0="./results/stylized.jpg",
    path1="./results/style.jpg",
    device="cuda",
)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vgg19 = VGG19().to(device)
    model = StylizingNetwork(ACTIAVTION).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=True)
    vgg19.eval()
    model.eval()

    # Load images
    print("Loading images...")
    contents_styles = list()
    for content_path, style_path in CONTENT_STYLE_PAIR:
        c = Image.open(content_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        c = toTensor255(c).unsqueeze(0).to(device)
        s = Image.open(style_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        s = toTensor255(s).unsqueeze(0).to(device)
        contents_styles.append([c, s, content_path, style_path])

    result = list()
    for i, (c, s, content_path, style_path) in enumerate(contents_styles):
        print(f"Processing {i + 1} ...")

        # Model inference
        with torch.no_grad():
            fc = vgg19(c)
            fs = vgg19(s)
            cs = model(fc, fs)
            cs = cs.clamp(0, 255)

        # Create output directory
        save_path = f"./results/{i + 1}"
        mkdir(save_path, delete_existing_files=True)

        # Save the results
        content_save_path = os.path.join(save_path, "content.png")
        style_save_path = os.path.join(save_path, "style.png")
        stylized_save_path = os.path.join(save_path, "stylized.png")
        toPil(c[0].byte()).save(content_save_path)
        toPil(s[0].byte()).save(style_save_path)
        toPil(cs[0].byte()).save(stylized_save_path)

        # Evaluate the results
        opt.path0 = stylized_save_path
        opt.path1 = content_save_path
        lpips_content = lpips_loss(opt, no_print=True)
        ssim_content = ssim_loss(opt, no_print=True)

        opt.path1 = style_save_path
        lpips_style = lpips_loss(opt, no_print=True)
        ssim_style = ssim_loss(opt, no_print=True)
        kl = kl_loss(opt, no_print=True)
        gram = gram_loss(opt, no_print=True)
        moment = nth_order_moment(opt, no_print=True)
        uni = uniformity(opt, no_print=True)
        entropy = average_entropy(opt, no_print=True)

        # Append the results
        result.append(
            {
                "content": content_path,
                "style": style_path,
                "lpips_content": lpips_content,
                "ssim_content": ssim_content,
                "lpips_style": lpips_style,
                "ssim_style": ssim_style,
                "kl": kl,
                "gram": gram,
                "moment": moment,
                "uniformity": uni,
                "entropy": entropy,
            }
        )

    # Calculate average results
    avg_result = np.mean([list(row.values())[2:] for row in result], axis=0)
    result.append(
        {
            "content": "average",
            "style": "average",
            "lpips_content": avg_result[0],
            "ssim_content": avg_result[1],
            "lpips_style": avg_result[2],
            "ssim_style": avg_result[3],
            "kl": avg_result[4],
            "gram": avg_result[5],
            "moment": avg_result[6],
            "uniformity": avg_result[7],
            "entropy": avg_result[8],
        }
    )

    # Save the results to a CSV file
    with open("./results/results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "content",
            "style",
            "lpips_content",
            "ssim_content",
            "lpips_style",
            "ssim_style",
            "kl",
            "gram",
            "moment",
            "uniformity",
            "entropy",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in result:
            writer.writerow(row)
    print("Results saved to ./results/results.csv")

import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import warnings

# خاموش کردن هشدارهای اضافی
warnings.filterwarnings("ignore")

def run_inference(model_path, image_path, conf_threshold):
    print(f"🧠 Loading model from '{model_path}'...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"🖼️ Reading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Cannot find image at {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("🔍 Running prediction...")
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # پس‌پردازش برای استخراج ماسک‌ها
    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image_rgb.shape[:2]],
        threshold=conf_threshold
    )[0]

    predicted_mask = results["segmentation"].cpu().numpy()
    found_cells = len(np.unique(predicted_mask)) - 1
    
    print(f"✅ Detection complete! Found {found_cells} nuclei instances.")

    # رسم و ذخیره تصویر
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(image_rgb)
    axs[0].set_title("Original Image", fontsize=14)
    axs[0].axis("off")

    axs[1].imshow(image_rgb)
    axs[1].imshow(predicted_mask, cmap="jet", alpha=0.4)
    axs[1].set_title(f"Predicted Nuclei (Count: {found_cells})", fontsize=14)
    axs[1].axis("off")

    plt.tight_layout()
    output_filename = "inference_output.png"
    plt.savefig(output_filename)
    print(f"💾 Result saved successfully as '{output_filename}'.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask2Former Inference for Cell Nuclei")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--source", type=str, required=True, help="Path to the input image")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    args = parser.parse_args()

    run_inference(args.model_dir, args.source, args.conf)
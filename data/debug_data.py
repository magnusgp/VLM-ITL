from datasets import load_dataset
import matplotlib.pyplot as plt

# 1) load the raw VOC-2012 train split
ds = load_dataset("nateraw/pascal-voc-2012", split="train")

# 2) pick the first 5 examples
examples = [ds[i] for i in range(5)]

# 3) plot them
fig, axes = plt.subplots(5, 2, figsize=(12, 20))
for row, ex in enumerate(examples):
    # raw RGB image
    img = ex["image"]           # PIL.Image
    axes[row,0].imshow(img)
    axes[row,0].axis("off")
    axes[row,0].set_title(f"Example {row} RGB") 

    # raw segmentation mask
    # HF stores it under annotation.segmentation_mask
    mask = ex["mask"]
    # convert palette→RGB so colors actually show
    mask_rgb = mask.convert("RGB")
    axes[row,1].imshow(mask_rgb)
    axes[row,1].axis("off")
    axes[row,1].set_title(f"Labels: "
        f"{sorted(set(mask.getdata()))[:20]}")
    print(f"Example {row} labels: "
          f"{sorted(set(mask.getdata()))[:20]}")
plt.tight_layout()
# 4) save the figure
fig.savefig("debug_data.png", dpi=300)
# 5) clsoe the figure
plt.close(fig)
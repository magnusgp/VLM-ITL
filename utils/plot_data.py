import numpy as np
import matplotlib.pyplot as plt

def debug_log_and_plot(images, masks, class_names, out_path="debug_samples.png"):
    """
    images:       tensor [B,3,H,W]  in [0..1] or [0..255]
    masks:        tensor [B,H,W]    ints in [0..n_classes-1]
    class_names:  list[str]         length == n_classes
    """
    B = images.size(0)
    n = min(5, B)
    # — console log —
    for i in range(n):
        m = masks[i].cpu().numpy()
        labels = np.unique(m)
        names  = [class_names[l] for l in labels]
        print(f" ▶ Sample {i:2d}: mask-IDs={labels.tolist()} → classes={names}")

    # — build figure —
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    for i in range(n):
        img = images[i].detach().cpu().permute(1,2,0).numpy()
        if img.max() > 1:
            img = img.astype(np.uint8)
        axes[0,i].imshow(img)
        axes[0,i].axis("off")
        axes[0,i].set_title(f"Image {i}")
        axes[1,i].imshow(masks[i].cpu().numpy(), cmap="tab20", vmin=0, vmax=len(class_names))
        axes[1,i].axis("off")
        axes[1,i].set_title(
            "\n".join([class_names[l] for l in np.unique(masks[i].cpu().numpy())])
        )
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\tWrote debug figure to {out_path}\n")

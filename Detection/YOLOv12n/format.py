import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to the main folder
main_folder = "XAI_Output"

# Number of subfolders and grid dimensions
num_images = 9
rows, cols = 3, 3  # 3x3 grid

# Collect all images
images = []
for i in range(1, num_images + 1):
    img_path = os.path.join(main_folder, f"image{i}", "limeCombined.png")
    if os.path.exists(img_path):
        images.append(Image.open(img_path))
    else:
        print(f"Image not found: {img_path}")

# Determine max width and height for each cell
widths, heights = zip(*(img.size for img in images))
max_width = max(widths)
max_height = max(heights)

# Create a figure
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))  # adjust size as needed

for idx, ax in enumerate(axs.flat):
    if idx < len(images):
        ax.imshow(images[idx])
    ax.axis('off')  # remove axes

plt.tight_layout()
plt.savefig("combined_limeImages.png", dpi=300, bbox_inches='tight')
plt.show()

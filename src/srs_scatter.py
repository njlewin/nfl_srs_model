import numpy as np
import pandas as pd
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from config import *




def get_logo(abbr: str, url: str, size: int = 64) -> Image.Image:
    path = CACHE / f"{abbr}.png"
    if not path.exists():
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    img = Image.open(path).convert("RGBA")
    img.thumbnail((size, size))  # keep aspect ratio, cap the size
    return img

CACHE.mkdir(exist_ok=True)
srs  = pd.read_csv(SRS_DATA_FILE)

# Plot OSRS vs DSRS with logos as scatter points.
fig, ax = plt.subplots(figsize=(8, 6))

# Invisible scatter so matplotlib sets the axis limits to fit the data
# (AnnotationBbox does not autoscale the axes on its own).
ax.scatter(srs["dsrs"], srs["osrs"], alpha=0)

for _, row in srs.iterrows():
    logo = get_logo(row["team"], row["team_logo_espn"])
    imagebox = OffsetImage(logo, zoom=0.5)  # tweak zoom to resize markers
    ab = AnnotationBbox(imagebox, (row["dsrs"], row["osrs"]), frameon=False)
    ab.set_zorder(3)  # logos on top of grid lines
    ax.add_artist(ab)

# Pad the limits out to whole numbers so the grid lines frame the data
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(int(xmin) - 1, int(xmax) + 1)
ax.set_ylim(int(ymin) - 1, int(ymax) + 1)

# Dotted grid line on every whole number
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(True, linestyle=":", linewidth=0.8, color="gray")
ax.set_axisbelow(True)

# Solid lines at 0 on both axes
ax.axhline(0, color="black", linestyle="-", linewidth=1.2, zorder=2)
ax.axvline(0, color="black", linestyle="-", linewidth=1.2, zorder=2)

season = srs["season"].iloc[0]
ax.set_xlabel("Defensive SRS (DSRS)\n Points prevented compared to an average defense. ")
ax.set_ylabel("Offensive SRS (OSRS)\n Points scored above an average offense.")
ax.set_title(f"{season} NFL Offensive vs Defensive SRS")

plt.tight_layout()
plt.savefig(ASSET_DIR + "nfl_srs_logo_scatter.png", dpi=200)
plt.show()
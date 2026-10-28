"""Replace only the title strip using real font glyphs; preserve the body."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops

root = Path("/Users/apple/Documents/Codex/2026-10-18/wo")
source = root / "outputs/sae_figures/Figure1B_title_only_v13.png"
target = root / "outputs/sae_figures/Figure1B_Arial_title_v14.png"
im = Image.open(source).convert("RGB")
width, height = im.size
strip_height, scale = 60, 4
title = "Sparse Reconstruction for Embedding Refinement"
font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
font = ImageFont.truetype(font_path, 40 * scale)
strip = Image.new("RGB", (width * scale, strip_height * scale), "white")
draw = ImageDraw.Draw(strip)
bbox = draw.textbbox((0, 0), title, font=font)
x = (width * scale - (bbox[2] - bbox[0])) / 2 - bbox[0]
y = 10 * scale - bbox[1]
draw.text((x, y), title, font=font, fill="#20344A")
strip = strip.resize((width, strip_height), Image.Resampling.LANCZOS)
result = im.copy()
result.paste(strip, (0, 0))
result.save(target)
assert result.size == im.size
assert ImageChops.difference(
    result.crop((0, strip_height, width, height)),
    im.crop((0, strip_height, width, height)),
).getbbox() is None
print(f"Saved {target}; {width}x{height}; all pixels below y=60 unchanged.")

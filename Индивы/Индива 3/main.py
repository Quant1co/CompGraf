from PIL import Image
import numpy as np
from pathlib import Path

size = 256
img = np.zeros((size, size), dtype=np.uint8)

# Создаём холмистый ландшафт с помощью синусов
for y in range(size):
    for x in range(size):
        # Плавные холмы
        h = 128  # базовый уровень (серый)
        h += int(30 * np.sin(x * 0.05) * np.cos(y * 0.05))  # большие холмы
        h += int(15 * np.sin(x * 0.1 + y * 0.1))  # средние волны
        h += int(10 * np.sin(x * 0.2) * np.sin(y * 0.15))  # мелкие детали
        img[y, x] = max(0, min(255, h))

output_dir = Path(r"C:\Users\Rodion\Documents\GitHub\CompGraf\Индивы\Индива 3\x64\Debug")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "heightmap.png"

Image.fromarray(img).save(output_path)
print(f"Saved: {output_path}")
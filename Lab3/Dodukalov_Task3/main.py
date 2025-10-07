from PIL import Image
import numpy as np

def barycentric_coords(p, a, b, c):
  
    v0 = b - a
    v1 = c - a
    v2 = p - a
    denom = v0[0]*v1[1] - v1[0]*v0[1]
    if abs(denom) < 1e-8:
        return None
    inv = 1.0 / denom
    u = (v2[0]*v1[1] - v1[0]*v2[1]) * inv
    v = (v0[0]*v2[1] - v2[0]*v0[1]) * inv
    w = 1 - u - v
    return np.array([w, u, v])

def rasterize_triangle(img_size, verts, colors):
 
    w, h = img_size
    out = np.ones((h, w, 3), dtype=np.uint8) * 255 

    a, b, c = np.array(verts[0]), np.array(verts[1]), np.array(verts[2])

    min_x = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
    max_x = min(int(np.ceil (max(a[0], b[0], c[0]))), w-1)
    min_y = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
    max_y = min(int(np.ceil (max(a[1], b[1], c[1]))), h-1)

    for yy in range(min_y, max_y+1):
        for xx in range(min_x, max_x+1):
            p = np.array([xx + 0.5, yy + 0.5])
            bary = barycentric_coords(p, a, b, c)
            if bary is None:
                continue
            if (bary >= -1e-6).all():
                col = bary[0]*colors[0] + bary[1]*colors[1] + bary[2]*colors[2]
                out[yy, xx] = np.clip(col, 0, 255).astype(np.uint8)
    return out

if __name__ == "__main__":
    width, height = 600, 400

    verts = np.array([
        [100, 50],  
        [500, 100],  
        [300, 350],  
    ], dtype=float)

    colors = np.array([
        [255, 0, 0],  
        [0, 255, 0],  
        [0, 0, 255],  
    ], dtype=float)

    img_arr = rasterize_triangle((width, height), verts, colors)
    img = Image.fromarray(img_arr, mode='RGB')

    img.save("triangle_gradient.png")
    img.show()

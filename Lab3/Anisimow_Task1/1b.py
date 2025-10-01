import tkinter as tk
from PIL import Image, ImageTk

# === Настройки ===
WIDTH, HEIGHT = 600, 600
DRAW_COLOR = "black"  # цвет для рисования границы
FILL_BORDER = (0, 0, 0)  # чёрная граница в RGB

# Загружаем текстуру
texture = Image.open(r"D:\CompGraf\Lab3\Anisimow_Task1\water.jpg").convert("RGB")
tex_width, tex_height = texture.size
texture_pixels = texture.load()

# Определяем, является ли текстура "маленькой" (для циклической заливки)
is_small_texture = tex_width <= WIDTH and tex_height <= HEIGHT
print(f"Текстура {'маленькая (циклическая)' if is_small_texture else 'большая (без цикла)'}")

# Создаём окно
root = tk.Tk()
root.title("Заливка текстурой")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

# Создаём холст как изображение
image = Image.new("RGB", (WIDTH, HEIGHT), "white")
pixels = image.load()
img_tk = ImageTk.PhotoImage(image)
canvas_img = canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

drawing = False
last_x, last_y = None, None

# --- Рисование границы ---
def start_draw(event):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    if drawing:
        canvas.create_line(last_x, last_y, event.x, event.y, fill=DRAW_COLOR, width=2)
        # прорисовываем линию и на PIL-изображении
        line_pixels(last_x, last_y, event.x, event.y, FILL_BORDER)
        last_x, last_y = event.x, event.y

def stop_draw(event):
    global drawing
    drawing = False

# Функция отрисовки линии (Брезенхем для пикселей на изображении)
def line_pixels(x1, y1, x2, y2, color):
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    while True:
        if 0 <= x1 < WIDTH and 0 <= y1 < HEIGHT:
            pixels[x1, y1] = color
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy

# --- Заливка текстурой (рекурсивная, на основе серий пикселов) ---
def flood_fill_texture_recursive(x, y, target_color, visited, start_x, start_y):
    # Проверка границ холста и посещённых пикселей
    if (x, y) in visited or x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return
    if pixels[x, y] != target_color:
        return
    visited.add((x, y))

    # Сканируем горизонтальную линию (серию пикселов)
    left = x
    while left > 0 and pixels[left - 1, y] == target_color and (left - 1, y) not in visited:
        left -= 1
    right = x
    while right < WIDTH - 1 and pixels[right + 1, y] == target_color and (right + 1, y) not in visited:
        right += 1

    # Заливаем линию текстурой
    for px in range(left, right + 1):
        tx_base = px - start_x
        ty_base = y - start_y
        
        if is_small_texture:
            tx = tx_base % tex_width
            ty = ty_base % tex_height
        else:
            tx = tx_base + (tex_width // 2)
            ty = ty_base + (tex_height // 2)
            tx = max(0, min(tex_width - 1, tx))
            ty = max(0, min(tex_height - 1, ty))
        
        pixels[px, y] = texture_pixels[tx, ty]
        visited.add((px, y))

    # Рекурсивно проверяем линии сверху и снизу
    for px in range(left, right + 1):
        if y - 1 >= 0:
            flood_fill_texture_recursive(px, y - 1, target_color, visited, start_x, start_y)
        if y + 1 < HEIGHT:
            flood_fill_texture_recursive(px, y + 1, target_color, visited, start_x, start_y)

def on_click_fill(event):
    """Запуск заливки по клику"""
    x, y = event.x, event.y
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        print("Клик за пределами холста")
        return
    target_color = pixels[x, y]

    # Не заливаем, если клик по границе
    if target_color == FILL_BORDER:
        print("Клик на границе, заливка не начата")
        return

    visited = set()
    flood_fill_texture_recursive(x, y, target_color, visited, x, y)

    # Обновляем картинку на холсте
    global img_tk
    img_tk = ImageTk.PhotoImage(image)
    canvas.itemconfig(canvas_img, image=img_tk)

    # Сохраняем холст для отладки
    image.save("filled_area.png")
    print("Холст сохранён в filled_area.png для проверки")

# Привязки событий
canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)
canvas.bind("<Button-3>", on_click_fill)  # правая кнопка мыши — заливка

root.mainloop()
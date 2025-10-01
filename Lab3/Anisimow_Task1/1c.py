import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math
import sys
import locale

# Установка кодировки UTF-8 для вывода в консоль
sys.stdout.reconfigure(encoding='utf-8')
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class AppConfig:
    def __init__(self):
        self.bg_color = "#2A2A2A"  # Темный фон для приложения

class BoundaryTracerApp:
    def __init__(self, window: tk.Tk, config):
        self.window = window
        self.config = config
        self.window.configure(bg=config.bg_color)
        self.visited_pixels = set()
        self.boundary_color = "#33AABB"  # Бирюзовый для границы
        self.is_tracing = False  # Флаг для предотвращения повторных вызовов

        self.window.title("Boundary Tracing")

        self.loaded_image = None

        # Кнопка для загрузки изображения
        self.load_button = tk.Button(
            window,
            text="Load Image",
            command=self.load_image,
            bg="#4CAF50",  # Зеленый фон
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            relief=tk.RAISED,
            borderwidth=3
        )
        self.load_button.pack(pady=10)

        self.image_canvas = tk.Canvas(self.window, bg=config.bg_color)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Button-1>", self.trace_boundary)

    def load_image(self):
        """Загружает и отображает изображение"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if file_path:
            self.image_path = file_path
            self.loaded_image = Image.open(file_path)

            # Ограничение размера изображения
            max_dimension = 600
            if self.loaded_image.width > max_dimension or self.loaded_image.height > max_dimension:
                scale = min(max_dimension / self.loaded_image.width, max_dimension / self.loaded_image.height)
                new_width = int(self.loaded_image.width * scale)
                new_height = int(self.loaded_image.height * scale)
                self.loaded_image = self.loaded_image.resize((new_width, new_height), Image.LANCZOS)

            self.img_width, self.img_height = self.loaded_image.size
            self.image_canvas.config(width=self.img_width, height=self.img_height)
            self.image_canvas.delete("all")
            self.display_image()

    def display_image(self):
        """Отображает изображение на холсте пиксель за пикселем"""
        for x in range(self.img_width):
            for y in range(self.img_height):
                rgb = self.loaded_image.getpixel((x, y))
                hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                self.image_canvas.create_rectangle(
                    x, y, x + 1, y + 1, fill=hex_color, outline=hex_color
                )

    def is_valid_pixel(self, x, y, target_color):
        """Проверяет, валиден ли пиксель и совпадает ли цвет точно"""
        if (
            (x, y) not in self.visited_pixels
            and 0 <= x < self.img_width
            and 0 <= y < self.img_height
        ):
            rgb = self.loaded_image.getpixel((x, y))
            pixel_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            return pixel_color == target_color
        return False

    def is_color_similar(self, x, y, target_color, tolerance=50):
        """Проверяет, близок ли цвет пикселя к целевому"""
        if (
            (x, y) not in self.visited_pixels
            and 0 <= x < self.img_width
            and 0 <= y < self.img_height
        ):
            rgb = self.loaded_image.getpixel((x, y))
            target_rgb = (
                int(target_color[1:3], 16),
                int(target_color[3:5], 16),
                int(target_color[5:7], 16)
            )
            color_distance = math.sqrt(
                sum((rgb[i] - target_rgb[i]) ** 2 for i in range(3))
            )
            return color_distance <= tolerance
        return False

    def is_edge_pixel(self, x, y, color):
        """Проверяет, является ли пиксель граничным"""
        neighbors = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.img_width and 0 <= ny < self.img_height:
                rgb = self.loaded_image.getpixel((nx, ny))
                neighbor_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                if neighbor_color != color:
                    return True
        return False

    def trace_boundary(self, event):
        """Обходит границу, начиная с точки щелчка"""
        if self.is_tracing:
            return  # Предотвращаем повторный запуск
        self.is_tracing = True
        self.visited_pixels.clear()  # Очищаем перед новым обходом

        x, y = event.x, event.y
        if not self.loaded_image:
            print("Please load an image first")
            self.is_tracing = False
            return

        rgb = self.loaded_image.getpixel((x, y))
        target_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

        if not self.is_edge_pixel(x, y, target_color):
            print("Selected point is not on the boundary")
            self.is_tracing = False
            return

        boundary_points = []
        neighbors = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        current_x, current_y = x, y

        if not self.is_color_similar(current_x, current_y, target_color):
            print("Point does not match boundary color")
            self.is_tracing = False
            return

        self.visited_pixels.add((current_x, current_y))
        boundary_points.append((current_x, current_y))

        current_dir = 4  # Начальное направление (запад)
        update_counter = 0

        while True:
            found_next = False
            search_dir = (current_dir + 2) % 8  # Поиск по часовой стрелке

            for i in range(8):
                d = (search_dir + i) % 8
                nx, ny = current_x + neighbors[d][0], current_y + neighbors[d][1]
                if self.is_color_similar(nx, ny, target_color) and self.is_edge_pixel(nx, ny, target_color):
                    boundary_points.append((nx, ny))
                    self.visited_pixels.add((nx, ny))
                    current_x, current_y = nx, ny
                    current_dir = (d + 4) % 8
                    found_next = True
                    update_counter += 1
                    if update_counter % 100 == 0:
                        self.window.update()
                    break

            if not found_next:
                break

            if current_x == x and current_y == y and len(boundary_points) > 2:
                break

        # Отрисовка границы
        for px, py in boundary_points:
            self.image_canvas.create_rectangle(
                px, py, px + 1, py + 1, fill=self.boundary_color, outline=self.boundary_color
            )

        print(f"Boundary found: {len(boundary_points)} points")
        self.is_tracing = False

def run_app():
    """Запускает приложение"""
    window = tk.Tk()
    config = AppConfig()
    app = BoundaryTracerApp(window, config)
    window.mainloop()

if __name__ == "__main__":
    run_app()
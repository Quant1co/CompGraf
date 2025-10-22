import tkinter as tk
import random

class FractalTerrainApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Симулятор Террейна")

        # Холст для отображения
        self.drawing_area = tk.Canvas(self.window, bg="white", width=800, height=400)
        self.drawing_area.pack(fill=tk.BOTH, expand=True)

        # Параметры конфигурации
        self.ruggedness_factor = tk.DoubleVar(value=1.0)
        self.recursion_depth = tk.IntVar(value=5)
        self.animation_pause = tk.IntVar(value=500)  # Интервал анимации в мс

        # Панель настроек
        settings_panel = tk.Frame(self.window)
        settings_panel.pack()

        tk.Label(settings_panel, text="Коэффициент неровности:").pack(side=tk.LEFT)
        self.ruggedness_slider = tk.Scale(settings_panel, from_=0.1, to=2.0, resolution=0.1,
                                          orient=tk.HORIZONTAL, variable=self.ruggedness_factor)
        self.ruggedness_slider.pack(side=tk.LEFT)

        tk.Label(settings_panel, text="Уровни рекурсии:").pack(side=tk.LEFT)
        self.depth_slider = tk.Scale(settings_panel, from_=1, to=10,
                                     orient=tk.HORIZONTAL, variable=self.recursion_depth)
        self.depth_slider.pack(side=tk.LEFT)

        tk.Label(settings_panel, text="Интервал анимации (мс):").pack(side=tk.LEFT)
        self.pause_slider = tk.Scale(settings_panel, from_=100, to=2000, orient=tk.HORIZONTAL,
                                     variable=self.animation_pause)
        self.pause_slider.pack(side=tk.LEFT)

        tk.Button(settings_panel, text="Создать ландшафт", command=self.initiate_creation).pack(side=tk.LEFT)
        tk.Button(settings_panel, text="Сброс", command=self.initiate_creation).pack(side=tk.LEFT)  # Кнопка Сброс

        # Label для текущего шага
        self.step_label = tk.Label(settings_panel, text="Шаг: 0 / 0")
        self.step_label.pack(side=tk.LEFT)

        self.drawing_area.bind("<Configure>", self.handle_resize)

        self.vertices = []  # Список вершин для отрисовки
        self.current_step = 0  # Для отслеживания шага
        self.resize_timer = None  # Для ресайза

    def handle_resize(self, event):
        # Debounce: отменяем предыдущий таймер и ставим новый
        if self.resize_timer:
            self.window.after_cancel(self.resize_timer)
        self.resize_timer = self.window.after(200, self.refresh_on_resize)  # Задержка 200 мс

    def refresh_on_resize(self):
        
        if not self.vertices:
            self.initiate_creation()  
            return

        old_width = self.vertices[-1][0] if self.vertices else 800  
        new_width = self.drawing_area.winfo_width()
        scale_x = new_width / old_width

        # Масштабируем X-координаты
        self.vertices = [(x * scale_x, y) for x, y in self.vertices]
        self.render_terrain()  

    def apply_midpoint_iteration(self, vertices, ruggedness):
        """Выполняет одну итерацию алгоритма и возвращает обновлённые вершины."""
        updated_vertices = []
        for i in range(len(vertices) - 1):
            pt1, pt2 = vertices[i], vertices[i + 1]
            avg_x = (pt1[0] + pt2[0]) / 2
            avg_y = (pt1[1] + pt2[1]) / 2

            # Случайное смещение с учётом фактора
            offset = random.uniform(-1, 1) * ruggedness
            avg_y += offset

            # Ограничение Y для предотвращения выхода за холст
            height = self.drawing_area.winfo_height()
            avg_y = max(50, min(height - 50, avg_y))  # Отступы сверху/снизу

            updated_vertices.append(pt1)
            updated_vertices.append((avg_x, avg_y))

        updated_vertices.append(vertices[-1])
        return updated_vertices

    def render_terrain(self):
        """Отрисовывает ландшафт на холсте."""
        self.drawing_area.delete("all")

        width = self.drawing_area.winfo_width()
        height = self.drawing_area.winfo_height()

        # Фон: градиент неба (рисуем один раз, оптимизировано)
        for y in range(height):
            blue_intensity = int(255 * (1 - y / height))
            color = f'#{blue_intensity:02x}{blue_intensity:02x}{255:02x}'
            self.drawing_area.create_line(0, y, width, y, fill=color)

        # Земля внизу (коричневый)
        self.drawing_area.create_rectangle(0, height - 50, width, height, fill="#8B4513", outline="")

        if len(self.vertices) < 2:
            return

        # Заполнение полигона под кривой (silhouette гор)
        poly_points = [(0, height)] + self.vertices + [(width, height)]
        self.drawing_area.create_polygon(poly_points, fill="#A9A9A9", outline="black", width=2)  # Серый для гор

        # Линии поверх (с изменением цвета по шагам)
        color_shade = f'#{min(255, 50 + self.current_step * 20):02x}00{255 - self.current_step * 20:02x}'  # От чёрного к синему
        for i in range(len(self.vertices) - 1):
            self.drawing_area.create_line(self.vertices[i][0], self.vertices[i][1],
                                          self.vertices[i + 1][0], self.vertices[i + 1][1],
                                          fill=color_shade, width=2)

    def perform_animation_phase(self, remaining_depth, current_ruggedness):
        """Анимирует фазы алгоритма с обновлением отрисовки."""
        if remaining_depth == 0:
            self.step_label.config(text="Шаг: Завершено")
            return

        self.current_step += 1
        total_depth = self.recursion_depth.get()
        self.step_label.config(text=f"Шаг: {self.current_step} / {total_depth}")

        # Применяем итерацию
        self.vertices = self.apply_midpoint_iteration(self.vertices, current_ruggedness)
        self.render_terrain()

        # Уменьшаем ruggedness и планируем следующую фазу
        self.window.after(self.animation_pause.get(), self.perform_animation_phase, remaining_depth - 1, current_ruggedness / 2)

    def initiate_creation(self):
        """Инициализирует вершины и запускает анимацию."""
        width = self.drawing_area.winfo_width()
        height = self.drawing_area.winfo_height()

        # Стартовые вершины 
        start_y = height // 2 + random.randint(-50, 50)
        end_y = height // 2 + random.randint(-50, 50)
        self.vertices = [(0, start_y), (width, end_y)]

        initial_ruggedness = self.ruggedness_factor.get() * (height // 2)
        depth = self.recursion_depth.get()

        self.current_step = 0
        self.step_label.config(text="Шаг: 0 / {}".format(depth))

        # Запуск анимации
        self.perform_animation_phase(depth, initial_ruggedness)

if __name__ == "__main__":
    main_window = tk.Tk()
    main_window.title("Лабораторная Работа")
    simulator = FractalTerrainApp(main_window)
    main_window.mainloop()
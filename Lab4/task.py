import pygame
import sys
import numpy as np
import math

# --- Матричные преобразования ---

def get_translation_matrix(dx, dy):
    """Возвращает матрицу смещения."""
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [dx, dy, 1]
    ])

def get_rotation_matrix(angle_degrees, center_x, center_y):
    """Возвращает матрицу поворота вокруг точки (center_x, center_y)."""
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Матрица для переноса центра в начало координат
    to_origin = get_translation_matrix(-center_x, -center_y)
    # Матрица поворота вокруг начала координат
    rotate = np.array([
        [cos_a,  sin_a, 0],
        [-sin_a, cos_a, 0],
        [0,      0,     1]
    ])
    # Матрица для переноса центра обратно
    from_origin = get_translation_matrix(center_x, center_y)
   
    # Комбинируем матрицы: перенос -> поворот -> обратный перенос
    return to_origin @ rotate @ from_origin

def get_scale_matrix(sx, sy, center_x, center_y):
    """Возвращает матрицу масштабирования относительно точки (center_x, center_y)."""
    # Матрица для переноса центра в начало координат
    to_origin = get_translation_matrix(-center_x, -center_y)
    # Матрица масштабирования
    scale = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ])
    # Матрица для переноса центра обратно
    from_origin = get_translation_matrix(center_x, center_y)
    
    # Комбинируем матрицы: перенос -> масштабирование -> обратный перенос
    return to_origin @ scale @ from_origin

# --- Класс полигона ---

class Polygon:
    def __init__(self):
        # Храним вершины как список списков для удобного добавления
        self.vertices = []

    def add_vertex(self, point):
        self.vertices.append(list(point)) # Преобразуем кортеж в список

    def draw(self, screen, color=(255, 0, 0), width=2):
        if len(self.vertices) == 1:
            pygame.draw.circle(screen, color, self.vertices[0], 3)
        elif len(self.vertices) >= 2:
            pygame.draw.lines(screen, color, True, self.vertices, width)

    def get_center(self):
        """Вычисляет геометрический центр (центроид) полигона."""
        if not self.vertices:
            return 0, 0
        x_coords = [p[0] for p in self.vertices]
        y_coords = [p[1] for p in self.vertices]
        center_x = sum(x_coords) / len(self.vertices)
        center_y = sum(y_coords) / len(self.vertices)
        return center_x, center_y

    def apply_transform(self, matrix):
        """Применяет матрицу преобразования ко всем вершинам полигона."""
        if not self.vertices:
            return
        
        # 1. Преобразуем наши 2D-вершины в однородные координаты (добавляем 1)
        # [[x1, y1], [x2, y2]] -> [[x1, y1, 1], [x2, y2, 1]]
        verts_np = np.array(self.vertices)
        ones = np.ones((verts_np.shape[0], 1))
        homogeneous_coords = np.hstack([verts_np, ones])

        # 2. Умножаем матрицу вершин на матрицу преобразования
        transformed_coords = homogeneous_coords @ matrix

        # 3. Преобразуем обратно в 2D-координаты и обновляем вершины
        # (Обычно w=1, но на всякий случай можно было бы делить на последний столбец)
        self.vertices = transformed_coords[:, :2].tolist()

    # --- Методы для конкретных преобразований ---

    def translate(self, dx, dy):
        """Смещение на dx, dy."""
        matrix = get_translation_matrix(dx, dy)
        self.apply_transform(matrix)

    def rotate(self, angle_degrees, pivot_point):
        """Поворот вокруг заданной пользователем точки."""
        cx, cy = pivot_point
        matrix = get_rotation_matrix(angle_degrees, cx, cy)
        self.apply_transform(matrix)

    def rotate_around_center(self, angle_degrees):
        """Поворот вокруг своего центра."""
        center = self.get_center()
        self.rotate(angle_degrees, center)

    def scale(self, sx, sy, pivot_point):
        """Масштабирование относительно заданной пользователем точки."""
        cx, cy = pivot_point
        matrix = get_scale_matrix(sx, sy, cx, cy)
        self.apply_transform(matrix)

    def scale_around_center(self, sx, sy):
        """Масштабирование относительно своего центра."""
        center = self.get_center()
        self.scale(sx, sy, center)


# --- Геометрические функции ---

def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    if (min(x1,x2) <= px <= max(x1,x2) and min(y1,y2) <= py <= max(y1,y2) and
        min(x3,x4) <= px <= max(x3,x4) and min(y3,y4) <= py <= max(y3,y4)):
        return (px, py)
    return None

def is_point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon.vertices)
    inside = False
    for i in range(n):
        x1, y1 = polygon.vertices[i]
        x2, y2 = polygon.vertices[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1)*(y - y1)/(y2 - y1) + x1):
            inside = not inside
    return inside

def classify_point_relative_to_edge(p1, p2, p):
    x1, y1 = p1
    x2, y2 = p2
    x, y = p

    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    if cross > 0:
        return 'слева'
    else:
        return 'справа'



# --- Основной класс приложения ---

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Lab 4: Affine Transformations")
        self.clock = pygame.time.Clock()
        self.polygons = []
        self.current_polygon = None
        self.running = True
        self.font = pygame.font.SysFont("Arial", 16)
        self.temp_edge = []
        self.intersection_point = None
        self.test_point = None
        self.point_status = ""
        self.edges = [] 

    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pos = pygame.mouse.get_pos()
                    if self.current_polygon is None:
                        self.current_polygon = Polygon()
                    self.current_polygon.add_vertex(pos)

                elif event.button == 3:
                    pos = pygame.mouse.get_pos()
                    self.temp_edge.append(pos)

                    if len(self.temp_edge) == 2:
                        self.edges.append((self.temp_edge[0], self.temp_edge[1]))

                    if len(self.temp_edge) == 4:
                        p1, p2, p3, p4 = self.temp_edge
                        self.intersection_point = line_intersection(p1, p2, p3, p4)
                        self.edges.append((p3, p4))
                        self.temp_edge = []

                elif event.button == 2:
                    pos = pygame.mouse.get_pos()
                    if self.polygons:
                        last_poly = self.polygons[-1]
                        inside = is_point_inside_polygon(pos, last_poly)
                        self.test_point = pos
                        self.point_status = "Внутри" if inside else "Снаружи"

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.current_polygon and len(self.current_polygon.vertices) > 0:
                        self.polygons.append(self.current_polygon)
                        self.current_polygon = None
                elif event.key == pygame.K_r:
                    self.polygons = []
                    self.current_polygon = None
                    self.intersection_point = None
                    self.test_point = None
                    self.temp_edge = []
                    self.edges = []  
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_l:
                    if len(self.edges) > 0:
                        p1, p2 = self.edges[-1]
                        p = pygame.mouse.get_pos()
                        side = classify_point_relative_to_edge(p1, p2, p)
                        self.point_status = f"Точка {side} от ребра"
                        self.test_point = p


                # --- Управление трансформациями для последнего полигона ---
                if self.polygons:
                    last_poly = self.polygons[-1]

                    # Смещение (стрелками)
                    if event.key == pygame.K_LEFT:
                        last_poly.translate(-10, 0)
                    elif event.key == pygame.K_RIGHT:
                        last_poly.translate(10, 0)
                    elif event.key == pygame.K_UP:
                        last_poly.translate(0, -10)
                    elif event.key == pygame.K_DOWN:
                        last_poly.translate(0, 10)

                    # Поворот вокруг своего центра (Q, E)    
                    elif event.key == pygame.K_q:
                        last_poly.rotate_around_center(-5)
                    elif event.key == pygame.K_e:
                        last_poly.rotate_around_center(5)

                    # Поворот вокруг курсора (A, D)    
                    elif event.key == pygame.K_a:
                        last_poly.rotate(-5, pygame.mouse.get_pos())
                    elif event.key == pygame.K_d:
                        last_poly.rotate(5, pygame.mouse.get_pos())

                    # Масштабирование отн. центра (+, -)    
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        last_poly.scale_around_center(1.1, 1.1)
                    elif event.key == pygame.K_MINUS:
                        last_poly.scale_around_center(0.9, 0.9)

                    # Масштабирование отн. курсора (W, S)    
                    elif event.key == pygame.K_w:
                        last_poly.scale(1.1, 1.1, pygame.mouse.get_pos())
                    elif event.key == pygame.K_s:
                        last_poly.scale(0.9, 0.9, pygame.mouse.get_pos())

    def draw_help_text(self):
        """Отрисовывает подсказку по управлению."""
        help_lines = [
            "--- Управление ---",
            "ЛКМ: Добавить вершину",
            "ПКМ: Добавить ребро (2 клика) ",
            "Средняя кнопка: Проверить точку внутри полигона",
            "L: Проверить точку относительно последнего ребра",
            "Пробел: Завершить полигон",
            "R: Очистить сцену",
            "ESC: Выход",
            "",
            "--- Трансформации ---",
            "Стрелки: Перемещение",
            "Q/E: Вращение вокруг центра",
            "A/D: Вращение вокруг курсора",
            "+/-: Масштаб отн. центра",
            "W/S: Масштаб отн. курсора",
        ]
        for i, line in enumerate(help_lines):
            text_surface = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 20))

    def draw(self):
        self.screen.fill((255, 255, 255))

        # Отрисовка всех полигонов
        for i, poly in enumerate(self.polygons):
            # Последний (выбранный) полигон рисуем другим цветом
            if i == len(self.polygons) - 1:
                poly.draw(self.screen, color=(0, 0, 255), width=3) # Синий и толще
            else:
                poly.draw(self.screen, color=(255, 0, 0)) # Красный
        
        # Отрисовка текущего создаваемого полигона
        if self.current_polygon:
            self.current_polygon.draw(self.screen, color=(0, 255, 0))

        # Отрисовка ребер 
        for e in self.edges:
            pygame.draw.line(self.screen, (0, 150, 255), e[0], e[1], 2)
            pygame.draw.circle(self.screen, (0, 200, 255), e[0], 4)
            pygame.draw.circle(self.screen, (0, 200, 255), e[1], 4)

        # Отображение точки пересечения
        if self.intersection_point:
            pygame.draw.circle(self.screen, (255, 0, 255),
                               (int(self.intersection_point[0]), int(self.intersection_point[1])), 6)

        # Отображение тестовой точки
        if self.test_point:
            color = (0, 200, 0) if "Внутри" in self.point_status else (200, 0, 0)
            pygame.draw.circle(self.screen, color, self.test_point, 5)
            text = self.font.render(self.point_status, True, (0, 0, 0))
            self.screen.blit(text, (self.test_point[0] + 10, self.test_point[1]))

        self.draw_help_text()
        pygame.display.flip()


if __name__ == "__main__":
    app = App()
    app.run()
    pygame.quit()
    sys.exit()

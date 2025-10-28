import pygame
import numpy as np
import math

# --- Классы ---

# Класс точки не нужен отдельно, так как мы будем использовать векторы NumPy
# Класс грани (многоугольника) не нужен отдельно, так как мы будем хранить индексы вершин

class Polyhedron:
    """
    Класс для представления многогранника.
    Хранит вершины в виде NumPy векторов и грани в виде списков индексов вершин.
    """
    def __init__(self, vertices, faces):
        """
        Инициализация многогранника.
        :param vertices: Список кортежей (x, y, z) для каждой вершины.
        :param faces: Список кортежей, где каждый кортеж содержит индексы вершин, образующих грань.
        """
        # Преобразуем вершины в однородные координаты [x, y, z, 1] для матричных преобразований
        self.vertices = np.array([list(v) + [1] for v in vertices], dtype=float)
        self.faces = faces
        self.color = (200, 200, 255)  # Цвет рёбер
        self.bg_color = (10, 20, 40)   # Цвет фона

    def apply_transform(self, matrix):
        """
        Применяет матрицу преобразования ко всем вершинам многогранника.
        """
        # np.dot(matrix, self.vertices.T).T можно заменить на self.vertices @ matrix.T
        self.vertices = self.vertices @ matrix.T

    def draw(self, surface, camera_distance, screen_width, screen_height):
        """
        Проецирует 3D точки на 2D экран и отрисовывает рёбра многогранника.
        """
        projected_points = []
        for vertex in self.vertices:
            x, y, z, _ = vertex
            
            # Простое перспективное проецирование
            # Чем дальше точка (больше z), тем меньше 'factor' и тем ближе к центру экрана она будет
            factor = camera_distance / (camera_distance + z)
            
            screen_x = int(x * factor + screen_width / 2)
            screen_y = int(y * factor + screen_height / 2)
            
            projected_points.append((screen_x, screen_y))
            
        # Отрисовка граней (многоугольников)
        for face in self.faces:
            points = [projected_points[i] for i in face]
            pygame.draw.polygon(surface, self.color, points, 2) # 2 - толщина линии


# --- Матрицы аффинных преобразований ---

def translation_matrix(tx, ty, tz):
    """Возвращает матрицу смещения."""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def scale_matrix(sx, sy, sz):
    """Возвращает матрицу масштабирования."""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def rotation_x_matrix(angle):
    """Возвращает матрицу поворота вокруг оси X."""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def rotation_y_matrix(angle):
    """Возвращает матрицу поворота вокруг оси Y."""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def rotation_z_matrix(angle):
    """Возвращает матрицу поворота вокруг оси Z."""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# --- Функции для создания многогранников ---

def create_tetrahedron(scale=100):
    vertices = np.array([
        (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
    ]) * scale / np.sqrt(3)
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    return Polyhedron(vertices, faces)

def create_hexahedron(scale=100): # Куб
    s = scale
    vertices = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
    ]
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 7, 4),
        (1, 2, 6, 5), (0, 1, 5, 4), (3, 2, 6, 7)
    ]
    return Polyhedron(vertices, faces)

def create_octahedron(scale=120):
    s = scale
    vertices = [
        (s, 0, 0), (-s, 0, 0), (0, s, 0), (0, -s, 0), (0, 0, s), (0, 0, -s)
    ]
    faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 2, 5), (1, 5, 3), (1, 3, 4), (1, 4, 2)
    ]
    return Polyhedron(vertices, faces)
    
def create_icosahedron(scale=120):
    phi = (1 + math.sqrt(5)) / 2 # Золотое сечение
    s = scale
    vertices = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
    ]
    vertices = np.array(vertices) * s / phi
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    return Polyhedron(vertices, faces)

def create_dodecahedron(scale=80):
    """Создает правильный додекаэдр (12 пятиугольных граней, 20 вершин)."""
    phi = (1 + math.sqrt(5)) / 2  # Золотое сечение
    s = scale
    
    # 20 вершин додекаэдра
    vertices = [
        # 8 вершин куба со сторонами 2
        ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
        (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1),
        # 12 вершин на прямоугольных гранях
        (0,  1/phi,  phi), (0,  1/phi, -phi), (0, -1/phi,  phi), (0, -1/phi, -phi),
        ( 1/phi,  phi, 0), ( 1/phi, -phi, 0), (-1/phi,  phi, 0), (-1/phi, -phi, 0),
        ( phi, 0,  1/phi), ( phi, 0, -1/phi), (-phi, 0,  1/phi), (-phi, 0, -1/phi)
    ]
    
    # Масштабируем вершины
    vertices = np.array([(v[0]*s, v[1]*s, v[2]*s) for v in vertices])
    
    # 12 пятиугольных граней додекаэдра
    faces = [
        (0, 16, 2, 10, 8),    # Грань 0
        (0, 8, 4, 14, 12),    # Грань 1
        (0, 12, 1, 17, 16),   # Грань 2
        (1, 9, 5, 14, 12),    # Грань 3
        (1, 17, 3, 11, 9),    # Грань 4
        (2, 16, 17, 3, 13),   # Грань 5
        (2, 13, 15, 6, 10),   # Грань 6
        (3, 11, 7, 15, 13),   # Грань 7
        (4, 8, 10, 6, 18),    # Грань 8
        (4, 18, 19, 5, 14),   # Грань 9
        (5, 19, 7, 11, 9),    # Грань 10
        (6, 15, 7, 19, 18)    # Грань 11
    ]
    
    return Polyhedron(vertices, faces)

# --- Основная часть программы ---

def draw_text(surface, text, pos, font, color=(255, 255, 255)):
    """Вспомогательная функция для отрисовки текста."""
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def main():
    pygame.init()
    
    screen_width, screen_height = 1000, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("3D Polyhedron Viewer")
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Consolas', 16)
    
    # Параметры
    camera_distance = 500
    rotation_speed = 1.0
    move_speed = 10.0
    scale_step = 1.05
    
    # Создаем первый многогранник
    polyhedrons = {
        '1': ('Тетраэдр', create_tetrahedron),
        '2': ('Гексаэдр (куб)', create_hexahedron),
        '3': ('Октаэдр', create_octahedron),
        '4': ('Икосаэдр', create_icosahedron),
        '5': ('Додекаэдр', create_dodecahedron),
    }
    current_poly_key = '2'
    polyhedron = polyhedrons[current_poly_key][1]()
    
    auto_rotate = {'x': False, 'y': True, 'z': False}
    
    running = True
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Смена фигур
                if '1' <= event.unicode <= '5':
                    current_poly_key = event.unicode
                    polyhedron = polyhedrons[current_poly_key][1]()
                # Сброс
                if event.key == pygame.K_r:
                    polyhedron = polyhedrons[current_poly_key][1]()
                
                # Включение/выключение авто-вращения
                if event.key == pygame.K_x: auto_rotate['x'] = not auto_rotate['x']
                if event.key == pygame.K_y: auto_rotate['y'] = not auto_rotate['y']
                if event.key == pygame.K_z: auto_rotate['z'] = not auto_rotate['z']


        # Обработка зажатых клавиш для преобразований
        keys = pygame.key.get_pressed()
        
        # Смещение
        if keys[pygame.K_UP]: 
            polyhedron.apply_transform(translation_matrix(0, -move_speed, 0))
        if keys[pygame.K_DOWN]: 
            polyhedron.apply_transform(translation_matrix(0, move_speed, 0))
        if keys[pygame.K_LEFT]: 
            polyhedron.apply_transform(translation_matrix(-move_speed, 0, 0))
        if keys[pygame.K_RIGHT]: 
            polyhedron.apply_transform(translation_matrix(move_speed, 0, 0))
        
        # Масштаб
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]: # Знак + (часто на той же клавише, что и =)
            polyhedron.apply_transform(scale_matrix(scale_step, scale_step, scale_step))
        if keys[pygame.K_MINUS]:
            polyhedron.apply_transform(scale_matrix(1/scale_step, 1/scale_step, 1/scale_step))
            
        # Поворот вручную (клавиши A, D, W, S, Q, E)
        if keys[pygame.K_d]:
            polyhedron.apply_transform(rotation_y_matrix(rotation_speed))
        if keys[pygame.K_a]:
            polyhedron.apply_transform(rotation_y_matrix(-rotation_speed))
        if keys[pygame.K_w]:
            polyhedron.apply_transform(rotation_x_matrix(rotation_speed))
        if keys[pygame.K_s]:
            polyhedron.apply_transform(rotation_x_matrix(-rotation_speed))
        if keys[pygame.K_e]:
            polyhedron.apply_transform(rotation_z_matrix(rotation_speed))
        if keys[pygame.K_q]:
            polyhedron.apply_transform(rotation_z_matrix(-rotation_speed))

        # Авто-вращение
        if auto_rotate['x']: polyhedron.apply_transform(rotation_x_matrix(rotation_speed / 2))
        if auto_rotate['y']: polyhedron.apply_transform(rotation_y_matrix(rotation_speed / 2))
        if auto_rotate['z']: polyhedron.apply_transform(rotation_z_matrix(rotation_speed / 2))

        # Отрисовка
        screen.fill(polyhedron.bg_color)
        polyhedron.draw(screen, camera_distance, screen_width, screen_height)
        
        # Отрисовка интерфейса
        info = [
            f"Текущая фигура: {polyhedrons[current_poly_key][0]}",
            " ",
            "Управление:",
            "Клавиши 1-5: Сменить фигуру",
            "Стрелки: Смещение (Translate)",
            "W/S: Поворот по X",
            "A/D: Поворот по Y",
            "Q/E: Поворот по Z",
            "+/-: Масштаб (Scale)",
            "R: Сброс положения фигуры",
            "X/Y/Z: Вкл/выкл авто-вращение",
        ]
        
        for i, line in enumerate(info):
            draw_text(screen, line, (10, 10 + i * 20), font)
        
        pygame.display.flip()
        
        clock.tick(60) # 60 FPS
        
    pygame.quit()


if __name__ == '__main__':
    main()
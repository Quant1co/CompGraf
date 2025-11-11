import pygame
import numpy as np
import math
from typing import List, Tuple

# --- Классы ---

class Point3D:
    """
    Класс для представления точки в 3D пространстве.
    """
    def __init__(self, x: float, y: float, z: float):
        """
        Инициализация точки в 3D пространстве.
        :param x: координата X
        :param y: координата Y
        :param z: координата Z
        """
        self.x = x
        self.y = y
        self.z = z
        # Храним точку как вектор в однородных координатах для матричных преобразований
        self.homogeneous = np.array([x, y, z, 1], dtype=float)
    
    def apply_transform(self, matrix: np.ndarray):
        """
        Применяет матрицу преобразования к точке.
        :param matrix: 4x4 матрица преобразования
        """
        self.homogeneous = self.homogeneous @ matrix.T
        self.x, self.y, self.z = self.homogeneous[:3]
    
    def project_perspective(self, camera_distance: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """
        Перспективная проекция.
        """
        factor = camera_distance / (camera_distance + self.z)
        screen_x = int(self.x * factor + screen_width / 2)
        screen_y = int(self.y * factor + screen_height / 2)
        return (screen_x, screen_y)
    
    def project_axonometric(self, view_matrix: np.ndarray, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """
        Аксонометрическая проекция.
        """
        v = self.homogeneous @ view_matrix.T
        screen_x = int(v[0] + screen_width / 2)
        screen_y = int(v[1] + screen_height / 2)
        return (screen_x, screen_y)
    
    def copy(self):
        """Создает копию точки."""
        return Point3D(self.x, self.y, self.z)
    
    def __repr__(self):
        return f"Point3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Polygon:
    """
    Класс для представления многоугольника (грани).
    """
    def __init__(self, vertex_indices: List[int], color: Tuple[int, int, int] = None):
        """
        Инициализация многоугольника.
        :param vertex_indices: список индексов вершин, образующих грань
        :param color: цвет грани (опционально)
        """
        self.vertex_indices = vertex_indices
        self.color = color if color else (200, 200, 255)
        self.fill_color = None  # Цвет заливки (если нужна заливка)
    
    def get_vertices(self, all_vertices: List[Point3D]) -> List[Point3D]:
        """
        Возвращает список вершин многоугольника.
        :param all_vertices: список всех вершин многогранника
        :return: список вершин данного многоугольника
        """
        return [all_vertices[i] for i in self.vertex_indices]
    
    def calculate_normal(self, vertices: List[Point3D]) -> np.ndarray:
        """
        Вычисляет нормаль к грани (для определения видимости).
        :param vertices: список вершин многогранника
        :return: вектор нормали
        """
        if len(self.vertex_indices) < 3:
            return np.array([0, 0, 1])
        
        # Берем первые три вершины для вычисления нормали
        p1 = vertices[self.vertex_indices[0]]
        p2 = vertices[self.vertex_indices[1]]
        p3 = vertices[self.vertex_indices[2]]
        
        # Векторы на грани
        v1 = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        v2 = np.array([p3.x - p1.x, p3.y - p1.y, p3.z - p1.z])
        
        # Векторное произведение дает нормаль
        normal = np.cross(v1, v2)
        return normal
    
    def draw(self, surface, projected_points: List[Tuple[int, int]], line_width: int = 2):
        """
        Отрисовка многоугольника на экране.
        :param surface: поверхность pygame для рисования
        :param projected_points: список спроецированных точек
        :param line_width: толщина линии
        """
        points = [projected_points[i] for i in self.vertex_indices]
        
        if self.fill_color:
            pygame.draw.polygon(surface, self.fill_color, points)
        
        pygame.draw.polygon(surface, self.color, points, line_width)
    
    def __repr__(self):
        return f"Polygon(vertices={self.vertex_indices})"


class Polyhedron:
    """
    Класс для представления многогранника.
    """
    def __init__(self, vertices: List[Tuple[float, float, float]], 
                 faces: List[Tuple[int, ...]], name: str = "Polyhedron"):
        """
        Инициализация многогранника.
        :param vertices: список кортежей (x, y, z) для каждой вершины
        :param faces: список кортежей с индексами вершин для каждой грани
        :param name: название многогранника
        """
        self.name = name
        self.vertices = [Point3D(x, y, z) for x, y, z in vertices]
        self.faces = [Polygon(list(face)) for face in faces]
        self.normalize_face_orientations()  # Нормализация ориентации граней
        self.edge_color = (200, 200, 255)
        self.bg_color = (10, 20, 40)
        self.show_faces = False  # Флаг для отображения заливки граней
    
    def normalize_face_orientations(self):
        """
        Нормализует ориентацию граней, чтобы все нормали были направлены наружу (dot > 0).
        """
        for face in self.faces:
            normal = face.calculate_normal(self.vertices)
            p = np.array([self.vertices[face.vertex_indices[0]].x, 
                          self.vertices[face.vertex_indices[0]].y, 
                          self.vertices[face.vertex_indices[0]].z])
            if np.dot(normal, p) < 0:
                face.vertex_indices.reverse()
    
    def apply_transform(self, matrix: np.ndarray):
        """
        Применяет матрицу преобразования ко всем вершинам многогранника.
        :param matrix: 4x4 матрица преобразования
        """
        for vertex in self.vertices:
            vertex.apply_transform(matrix)
    
    def get_center(self) -> Point3D:
        """
        Вычисляет центр многогранника.
        :return: точка центра
        """
        avg_x = sum(v.x for v in self.vertices) / len(self.vertices)
        avg_y = sum(v.y for v in self.vertices) / len(self.vertices)
        avg_z = sum(v.z for v in self.vertices) / len(self.vertices)
        return Point3D(avg_x, avg_y, avg_z)
    
    def draw(self, surface, camera_distance: float, screen_width: int, screen_height: int, projection_mode: str = 'perspective'):
        """
        Проецирует и отрисовывает многогранник.
        :param surface: поверхность pygame для рисования
        :param camera_distance: расстояние до камеры
        :param screen_width: ширина экрана
        :param screen_height: высота экрана
        :param projection_mode: режим проекции: 'perspective' или 'axonometric'
        """
        if projection_mode == 'axonometric':
            view = axonometric_view_matrix()
            # Вычисляем viewed вершины
            viewed_vertices = []
            for vertex in self.vertices:
                v_hom = np.array([vertex.x, vertex.y, vertex.z, 1.0])
                viewed_hom = v_hom @ view.T
                viewed_vertices.append(Point3D(viewed_hom[0], viewed_hom[1], viewed_hom[2]))
        else:
            viewed_vertices = self.vertices
        
        # Проецируем вершины
        projected_points = []
        if projection_mode == 'perspective':
            for vertex in self.vertices:
                projected_points.append(vertex.project_perspective(camera_distance, screen_width, screen_height))
        else:  # axonometric
            for viewed_vertex in viewed_vertices:
                projected_points.append((int(viewed_vertex.x + screen_width / 2), int(viewed_vertex.y + screen_height / 2)))
        
        # Сортируем грани по глубине
        face_depths = []
        for face in self.faces:
            avg_z = sum(viewed_vertices[i].z for i in face.vertex_indices) / len(face.vertex_indices)
            face_depths.append((face, avg_z))
        
        # Сортируем от дальних к ближним (assuming z increases into the screen)
        face_depths.sort(key=lambda x: x[1], reverse=True)
        
        # Отрисовываем грани
        for face, _ in face_depths:
            # Backface culling
            normal = face.calculate_normal(viewed_vertices)
            if normal[2] < 0:  # Грань повернута к нам
                face.draw(surface, projected_points)
    
    def get_info(self) -> str:
        """
        Возвращает информацию о многограннике.
        :return: строка с информацией
        """
        return f"{self.name}: {len(self.vertices)} вершин, {len(self.faces)} граней"
    
    def __repr__(self):
        return f"Polyhedron(name={self.name}, vertices={len(self.vertices)}, faces={len(self.faces)})"


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
    """Возвращает матрицу поворота вокруг оси X (угол в градусах)."""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def rotation_y_matrix(angle):
    """Возвращает матрицу поворота вокруг оси Y (угол в градусах)."""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def rotation_z_matrix(angle):
    """Возвращает матрицу поворота вокруг оси Z (угол в градусах)."""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def reflection_matrix(plane: str):
    """
    Возвращает матрицу отражения относительно выбранной координатной плоскости.
    :param plane: 'xy' (отражение по Z), 'xz' (по Y), 'yz' (по X)
    """
    if plane == 'xy':
        return scale_matrix(1, 1, -1)
    elif plane == 'xz':
        return scale_matrix(1, -1, 1)
    elif plane == 'yz':
        return scale_matrix(-1, 1, 1)
    else:
        raise ValueError("Неверная плоскость для отражения")

# --- Новые матрицы: поворот вокруг произвольной прямой ---

def rodrigues_rotation_matrix(u: np.ndarray, angle_rad: float) -> np.ndarray:
    """Возвращает 3x3 матрицу поворота по формуле Родригеса для единичного вектора u."""
    ux, uy, uz = u
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    I = np.eye(3)
    uu = np.outer(u, u)
    u_x = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    R = c * I + (1 - c) * uu + s * u_x
    return R


def rotation_about_line(p1: Tuple[float, float, float], p2: Tuple[float, float, float], angle_deg: float) -> np.ndarray:
    """
    Возвращает 4x4 матрицу поворота вокруг прямой, проходящей через p1 и p2, на угол angle_deg (в градусах).
    Алгоритм: перенести p1 в начало координат, применить поворот по Родригесу вокруг единичного направления u, вернуть обратно.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    u = p2 - p1
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("Две точки совпадают — направление неопределено")
    u = u / norm
    angle_rad = math.radians(angle_deg)
    R3 = rodrigues_rotation_matrix(u, angle_rad)
    # Собираем однородную матрицу
    R = np.eye(4)
    R[:3, :3] = R3
    T1 = translation_matrix(-p1[0], -p1[1], -p1[2])
    T2 = translation_matrix(p1[0], p1[1], p1[2])
    return T2 @ R @ T1


def rotation_axis_through_center_matrix(center: Point3D, axis: str, angle_deg: float) -> np.ndarray:
    """
    Возвращает 4x4 матрицу поворота вокруг прямой, проходящей через центр и параллельной оси axis ('x','y','z').
    """
    if axis == 'x':
        R = rotation_x_matrix(angle_deg)
    elif axis == 'y':
        R = rotation_y_matrix(angle_deg)
    elif axis == 'z':
        R = rotation_z_matrix(angle_deg)
    else:
        raise ValueError("Неверная ось")
    trans_to_origin = translation_matrix(-center.x, -center.y, -center.z)
    trans_back = translation_matrix(center.x, center.y, center.z)
    return trans_back @ R @ trans_to_origin

# --- Аксонометрическая view-матрица ---

def axonometric_view_matrix():
    """
    Возвращает 4x4 матрицу, задающую стандартную изометрическую аксонометрию.
    Комбинация: поворот вокруг X на 35.264° и вокруг Y на 45° (в таком порядке).
    """
    Rx = rotation_x_matrix(35.26438968)  # ~arctan(sqrt(1/2)) в градусах
    Ry = rotation_y_matrix(45)
    return Ry @ Rx

# --- Функции для создания многогранников ---

def create_tetrahedron(scale=100):
    """Создает правильный тетраэдр."""
    vertices = np.array([
        (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
    ]) * scale / np.sqrt(3)
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    return Polyhedron(vertices.tolist(), faces, "Тетраэдр")

def create_hexahedron(scale=100):
    """Создает куб."""
    s = scale
    vertices = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
    ]
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 7, 4),
        (1, 2, 6, 5), (0, 1, 5, 4), (3, 2, 6, 7)
    ]
    return Polyhedron(vertices, faces, "Гексаэдр (Куб)")

def create_octahedron(scale=120):
    """Создает правильный октаэдр."""
    s = scale
    vertices = [
        (s, 0, 0), (-s, 0, 0), (0, s, 0), (0, -s, 0), (0, 0, s), (0, 0, -s)
    ]
    faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 2, 5), (1, 5, 3), (1, 3, 4), (1, 4, 2)
    ]
    return Polyhedron(vertices, faces, "Октаэдр")

def create_icosahedron(scale=120):
    """Создает правильный икосаэдр."""
    phi = (1 + math.sqrt(5)) / 2
    s = scale
    vertices = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
    ]
    vertices = (np.array(vertices) * s / phi).tolist()
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    return Polyhedron(vertices, faces, "Икосаэдр")

def create_dodecahedron(scale=80):
    """Создает правильный додекаэдр."""
    phi = (1 + math.sqrt(5)) / 2
    s = scale
    
    vertices = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        (0, 1/phi, phi), (0, 1/phi, -phi), (0, -1/phi, phi), (0, -1/phi, -phi),
        (1/phi, phi, 0), (1/phi, -phi, 0), (-1/phi, phi, 0), (-1/phi, -phi, 0),
        (phi, 0, 1/phi), (phi, 0, -1/phi), (-phi, 0, 1/phi), (-phi, 0, -1/phi)
    ]
    
    vertices = [(v[0]*s, v[1]*s, v[2]*s) for v in vertices]
    
    faces = [
        (0, 16, 2, 10, 8), (0, 8, 4, 14, 12), (0, 12, 1, 17, 16),
        (1, 9, 5, 14, 12), (1, 17, 3, 11, 9), (2, 16, 17, 3, 13),
        (2, 13, 15, 6, 10), (3, 11, 7, 15, 13), (4, 8, 10, 6, 18),
        (4, 18, 19, 5, 14), (5, 19, 7, 11, 9), (6, 15, 7, 19, 18)
    ]
    
    return Polyhedron(vertices, faces, "Додекаэдр")

# --- Основная часть программы ---

def draw_text(surface, text, pos, font, color=(255, 255, 255)):
    """Вспомогательная функция для отрисовки текста."""
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def main():
    pygame.init()
    
    screen_width, screen_height = 1000, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("3D Polyhedron Viewer - Extended")
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Consolas', 16)
    
    # Параметры
    camera_distance = 500
    rotation_speed = 1.0
    move_speed = 10.0
    scale_step = 1.05
    
    # Параметры для поворота вокруг произвольной прямой (можно менять в коде)
    arbitrary_p1 = (0.0, 0.0, 0.0)
    arbitrary_p2 = (100.0, 100.0, 0.0)
    arbitrary_angle_step = 15.0  # градусов за одно нажатие клавиши 'k'
    
    # Создаем многогранники
    polyhedrons = {
        '1': create_tetrahedron,
        '2': create_hexahedron,
        '3': create_octahedron,
        '4': create_icosahedron,
        '5': create_dodecahedron,
    }
    
    current_poly_key = '2'
    polyhedron = polyhedrons[current_poly_key]()
    
    auto_rotate = {'x': False, 'y': True, 'z': False}
    projection_mode = 'perspective'  # 'perspective' или 'axonometric'
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Смена фигур
                if '1' <= event.unicode <= '5':
                    current_poly_key = event.unicode
                    polyhedron = polyhedrons[current_poly_key]()
                
                # Сброс
                if event.key == pygame.K_r:
                    polyhedron = polyhedrons[current_poly_key]()
                
                # Авто-вращение
                if event.key == pygame.K_x: 
                    auto_rotate['x'] = not auto_rotate['x']
                if event.key == pygame.K_y: 
                    auto_rotate['y'] = not auto_rotate['y']
                if event.key == pygame.K_z: 
                    auto_rotate['z'] = not auto_rotate['z']
                
                # Отражения (применяются один раз при нажатии)
                if event.key == pygame.K_6:
                    polyhedron.apply_transform(reflection_matrix('xy'))  # Отражение относительно XY
                if event.key == pygame.K_7:
                    polyhedron.apply_transform(reflection_matrix('xz'))  # Отражение относительно XZ
                if event.key == pygame.K_8:
                    polyhedron.apply_transform(reflection_matrix('yz'))  # Отражение относительно YZ

                # Переключение проекций
                if event.key == pygame.K_p:
                    projection_mode = 'axonometric' if projection_mode == 'perspective' else 'perspective'

                # Поворот вокруг произвольной прямой (по умолчанию — arbitrary_p1->arbitrary_p2)
                if event.key == pygame.K_k:
                    M = rotation_about_line(arbitrary_p1, arbitrary_p2, arbitrary_angle_step)
                    polyhedron.apply_transform(M)

                # Повороты вокруг прямой, проходящей через центр, параллельно выбранной оси
                if event.key == pygame.K_u:  # вокруг оси X через центр
                    center = polyhedron.get_center()
                    M = rotation_axis_through_center_matrix(center, 'x', 10)
                    polyhedron.apply_transform(M)
                if event.key == pygame.K_i:  # вокруг оси Y через центр
                    center = polyhedron.get_center()
                    M = rotation_axis_through_center_matrix(center, 'y', 10)
                    polyhedron.apply_transform(M)
                if event.key == pygame.K_o:  # вокруг оси Z через центр
                    center = polyhedron.get_center()
                    M = rotation_axis_through_center_matrix(center, 'z', 10)
                    polyhedron.apply_transform(M)

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
        
        # Масштаб относительно центра
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            center = polyhedron.get_center()
            trans_to_origin = translation_matrix(-center.x, -center.y, -center.z)
            scale_mat = scale_matrix(scale_step, scale_step, scale_step)
            trans_back = translation_matrix(center.x, center.y, center.z)
            combined = trans_back @ scale_mat @ trans_to_origin
            polyhedron.apply_transform(combined)
        if keys[pygame.K_MINUS]:
            center = polyhedron.get_center()
            trans_to_origin = translation_matrix(-center.x, -center.y, -center.z)
            scale_mat = scale_matrix(1/scale_step, 1/scale_step, 1/scale_step)
            trans_back = translation_matrix(center.x, center.y, center.z)
            combined = trans_back @ scale_mat @ trans_to_origin
            polyhedron.apply_transform(combined)
        
        # Поворот вручную
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
        if auto_rotate['x']: 
            polyhedron.apply_transform(rotation_x_matrix(rotation_speed / 2))
        if auto_rotate['y']: 
            polyhedron.apply_transform(rotation_y_matrix(rotation_speed / 2))
        if auto_rotate['z']: 
            polyhedron.apply_transform(rotation_z_matrix(rotation_speed / 2))

        # Отрисовка
        screen.fill(polyhedron.bg_color)
        polyhedron.draw(screen, camera_distance, screen_width, screen_height, projection_mode)
        
        # Интерфейс
        info = [
            polyhedron.get_info(),
            f"Проекция: {projection_mode}",
            "",
            "Управление:",
            "1-5: Сменить фигуру",
            "Стрелки: Смещение",
            "W/S, A/D, Q/E: Поворот",
            "+/-: Масштаб (относительно центра)",
            "R: Сброс",
            "X/Y/Z: Авто-вращение",
            "6/7/8: Отражения относительно XY/XZ/YZ",
            "P: Переключить перспективу/аксонометрию",
            "U/I/O: Поворот вокруг прямой через центр (X/Y/Z) на 10°",
            "K: Поворот вокруг произвольной прямой (по умолчанию задается в коде)",
            f"Arb line p1={arbitrary_p1} p2={arbitrary_p2} step={arbitrary_angle_step}° (нажмите K)",
        ]
        
        for i, line in enumerate(info):
            draw_text(screen, line, (10, 10 + i * 20), font)
        
        # Показываем состояние авто-вращения
        auto_status = "Авто: "
        if auto_rotate['x']: auto_status += "X "
        if auto_rotate['y']: auto_status += "Y "
        if auto_rotate['z']: auto_status += "Z "
        if not any(auto_rotate.values()): auto_status += "Выкл"
        draw_text(screen, auto_status, (10, 10 + len(info) * 20), font, (100, 255, 100))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == '__main__':
    main()

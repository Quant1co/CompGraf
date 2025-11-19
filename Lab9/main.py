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
        # Защита от деления на ноль или ухода точки за камеру
        if self.z <= -camera_distance + 1:
            factor = 0
        else:
            factor = camera_distance / (camera_distance + self.z)
            
        screen_x = int(self.x * factor + screen_width / 2)
        screen_y = int(self.y * factor + screen_height / 2)
        return (screen_x, screen_y)
    
    def project_axonometric(self, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """
        Аксонометрическая проекция.
        """
        screen_x = int(self.x + screen_width / 2)
        screen_y = int(self.y + screen_height / 2)
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
    
    def draw(self, surface, projected_points: List[Tuple[int, int]], line_width: int = 1):
        """
        Отрисовка многоугольника на экране.
        :param surface: поверхность pygame для рисования
        :param projected_points: список спроецированных точек
        :param line_width: толщина линии
        """
        points = [projected_points[i] for i in self.vertex_indices]
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
        # Параметры освещения/шейдинга
        self.object_color = (180, 200, 255)
        self.light_color = (255, 255, 255)
        self.light_position = np.array([400.0, 400.0, 600.0, 1.0])  # в мировых координатах
        self.ambient_intensity = 0.2
        self.shading_mode = 'flat'  # 'flat' | 'gouraud' | 'phong_toon'
        # Опциональные нормали из OBJ (список np.ndarray длины == количеству вершин)
        self.obj_vertex_normals: List[np.ndarray] | None = None
    
    def normalize_face_orientations(self):
        """
        Нормализует ориентацию граней, чтобы все нормали были направлены наружу.
        """
        center = self.get_center()
        center_vec = np.array([center.x, center.y, center.z])
        
        for face in self.faces:
            normal = face.calculate_normal(self.vertices)
            if np.linalg.norm(normal) == 0:
                continue  # Пропустить вырожденные грани
            
            p = np.array([self.vertices[face.vertex_indices[0]].x, 
                          self.vertices[face.vertex_indices[0]].y, 
                          self.vertices[face.vertex_indices[0]].z])
            vector_to_p = p - center_vec
            if np.dot(normal, vector_to_p) < 0:
                face.vertex_indices.reverse()
                # Пересчитать нормаль после реверса
                normal = face.calculate_normal(self.vertices)
    
    def apply_transform(self, matrix: np.ndarray):
        """
        Применяет матрицу преобразования ко всем вершинам многогранника.
        :param matrix: 4x4 матрица преобразования
        """
        for vertex in self.vertices:
            vertex.apply_transform(matrix)
        if self.obj_vertex_normals is not None:
            try:
                normal_matrix = np.linalg.inv(matrix).T
            except np.linalg.LinAlgError:
                normal_matrix = None
            if normal_matrix is not None:
                for idx, normal in enumerate(self.obj_vertex_normals):
                    vec = np.array([normal[0], normal[1], normal[2], 0.0])
                    transformed = normal_matrix @ vec
                    n = transformed[:3]
                    length = np.linalg.norm(n)
                    self.obj_vertex_normals[idx] = n / length if length > 1e-8 else np.array([0.0, 0.0, 1.0])
        self.normalize_face_orientations()  # Обеспечиваем нормали наружу после трансформации
    
    def get_center(self) -> Point3D:
        """
        Вычисляет центр многогранника.
        :return: точка центра
        """
        if not self.vertices:
            return Point3D(0, 0, 0)
        avg_x = sum(v.x for v in self.vertices) / len(self.vertices)
        avg_y = sum(v.y for v in self.vertices) / len(self.vertices)
        avg_z = sum(v.z for v in self.vertices) / len(self.vertices)
        return Point3D(avg_x, avg_y, avg_z)
    
    def draw(self, surface, camera_distance: float, screen_width: int, screen_height: int, projection_mode: str = 'perspective', camera_rotation: np.ndarray = np.eye(4), z_buffer: np.ndarray = None):
        """
        Проецирует и отрисовывает многогранник с использованием Z-буфера.
        """
        if z_buffer is None:
            # Если z-буфер не предоставлен, используется старый метод отрисовки
            self.draw_legacy(surface, camera_distance, screen_width, screen_height, projection_mode, camera_rotation)
            return

        if projection_mode == 'axonometric':
            proj_matrix = axonometric_view_matrix()
        else:
            proj_matrix = np.eye(4)
        
        full_view = proj_matrix @ camera_rotation
        
        # Вычисляем viewed вершины
        viewed_vertices = []
        for vertex in self.vertices:
            v_hom = vertex.homogeneous @ full_view.T
            viewed = Point3D(v_hom[0], v_hom[1], v_hom[2])
            viewed_vertices.append(viewed)
        
        # Проецируем вершины
        if projection_mode == 'perspective':
            projected_points = [v.project_perspective(camera_distance, screen_width, screen_height) for v in viewed_vertices]
        else:  # axonometric
            projected_points = [v.project_axonometric(screen_width, screen_height) for v in viewed_vertices]

        # Позиция источника света в пространстве камеры/проекции
        light_view = None
        if self.light_position is not None:
            light_view = self.light_position @ full_view.T

        # Нормали вершин в пространстве обзора
        if self.obj_vertex_normals is not None:
            vertex_normals_view = self.transform_normals(self.obj_vertex_normals, full_view)
        else:
            vertex_normals_view = self.compute_vertex_normals(viewed_vertices)
        
        # Отрисовываем грани с использованием z-буфера
        for face in self.faces:
            # Backface culling
            normal = face.calculate_normal(viewed_vertices)
            if normal[2] > 1e-6:  # Грань отвернута от нас
                continue

            # Триангуляция грани (для простоты, предполагаем выпуклые многоугольники)
            if len(face.vertex_indices) < 3:
                continue
            
            # Цвет грани для плоского освещения
            face_color = (100, 100, 100)
            if self.shading_mode == 'flat':
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-6:
                    face_color = self.vertex_color_lambert(
                        viewed_vertices[face.vertex_indices[0]],
                        normal / norm_len,
                        light_view
                    )


            # Разбиваем на треугольники от первой вершины
            v0_idx = face.vertex_indices[0]
            for i in range(1, len(face.vertex_indices) - 1):
                v1_idx = face.vertex_indices[i]
                v2_idx = face.vertex_indices[i + 1]
                
                p0, p1, p2 = projected_points[v0_idx], projected_points[v1_idx], projected_points[v2_idx]
                z0, z1, z2 = viewed_vertices[v0_idx].z, viewed_vertices[v1_idx].z, viewed_vertices[v2_idx].z
                # Данные вершин для шейдинга
                if self.shading_mode == 'gouraud':
                    c0 = np.array(self.vertex_color_lambert(viewed_vertices[v0_idx], vertex_normals_view[v0_idx], light_view))
                    c1 = np.array(self.vertex_color_lambert(viewed_vertices[v1_idx], vertex_normals_view[v1_idx], light_view))
                    c2 = np.array(self.vertex_color_lambert(viewed_vertices[v2_idx], vertex_normals_view[v2_idx], light_view))
                elif self.shading_mode == 'phong_toon':
                    n0 = vertex_normals_view[v0_idx]
                    n1 = vertex_normals_view[v1_idx]
                    n2 = vertex_normals_view[v2_idx]
                    pv0 = np.array([viewed_vertices[v0_idx].x, viewed_vertices[v0_idx].y, viewed_vertices[v0_idx].z])
                    pv1 = np.array([viewed_vertices[v1_idx].x, viewed_vertices[v1_idx].y, viewed_vertices[v1_idx].z])
                    pv2 = np.array([viewed_vertices[v2_idx].x, viewed_vertices[v2_idx].y, viewed_vertices[v2_idx].z])

                # Ограничивающий прямоугольник для треугольника
                x_min = max(0, min(p0[0], p1[0], p2[0]))
                x_max = min(screen_width - 1, max(p0[0], p1[0], p2[0]))
                y_min = max(0, min(p0[1], p1[1], p2[1]))
                y_max = min(screen_height - 1, max(p0[1], p1[1], p2[1]))
                
                # Проверка размера bbox, чтобы не виснуть на больших гранях при глюках
                if (x_max - x_min) * (y_max - y_min) > 500000: 
                    continue

                # Проход по пикселям в ограничивающем прямоугольнике
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        coords = barycentric_coords((x, y), p0, p1, p2)
                        if coords is None: continue
                        
                        alpha, beta, gamma = coords
                        # Если точка внутри треугольника
                        if alpha >= 0 and beta >= 0 and gamma >= 0:
                            # Интерполируем z
                            interpolated_z = interpolate_z(coords, z0, z1, z2)
                            
                            # Проверка z-буфера
                            if interpolated_z < z_buffer[y, x]:
                                z_buffer[y, x] = interpolated_z
                                if self.shading_mode == 'flat':
                                    surface.set_at((x, y), face_color)
                                elif self.shading_mode == 'gouraud':
                                    col = alpha * c0 + beta * c1 + gamma * c2
                                    col = np.clip(col, 0, 255).astype(int)
                                    surface.set_at((x, y), tuple(col))
                                elif self.shading_mode == 'phong_toon':
                                    n_interp = alpha * n0 + beta * n1 + gamma * n2
                                    n_len = np.linalg.norm(n_interp)
                                    if n_len < 1e-6:
                                        n_hat = np.array([0.0, 0.0, 1.0])
                                    else:
                                        n_hat = n_interp / n_len
                                    pos_interp = alpha * pv0 + beta * pv1 + gamma * pv2
                                    pos_point = Point3D(pos_interp[0], pos_interp[1], pos_interp[2])
                                    color = self.phong_toon_color(pos_point, n_hat, light_view)
                                    surface.set_at((x, y), color)
            
            # Отрисовка рёбер поверх
            face.draw(surface, projected_points)

    def draw_legacy(self, surface, camera_distance: float, screen_width: int, screen_height: int, projection_mode: str = 'perspective', camera_rotation: np.ndarray = np.eye(4)):
        """
        Проецирует и отрисовывает многогранник (старый метод с сортировкой).
        """
        if projection_mode == 'axonometric':
            proj_matrix = axonometric_view_matrix()
        else:
            proj_matrix = np.eye(4)
        
        full_view = proj_matrix @ camera_rotation
        
        # Вычисляем viewed вершины
        viewed_vertices = []
        for vertex in self.vertices:
            v_hom = vertex.homogeneous @ full_view.T
            viewed = Point3D(v_hom[0], v_hom[1], v_hom[2])
            viewed_vertices.append(viewed)
        
        # Проецируем вершины
        if projection_mode == 'perspective':
            projected_points = [v.project_perspective(camera_distance, screen_width, screen_height) for v in viewed_vertices]
        else:  # axonometric
            projected_points = [v.project_axonometric(screen_width, screen_height) for v in viewed_vertices]
        
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
            if normal[2] < -1e-6:  # Грань повернута к нам (dot(normal, [0,0,1]) < 0)
                # Заливка грани
                points = [projected_points[i] for i in face.vertex_indices]
                pygame.draw.polygon(surface, (50, 50, 80), points) # Заливка цветом фона
                face.draw(surface, projected_points)
    
    
    def get_info(self) -> str:
        """
        Возвращает информацию о многограннике.
        :return: строка с информацией
        """
        return f"{self.name}: {len(self.vertices)} вершин, {len(self.faces)} граней"
    
    def to_obj(self, filename: str):
        """
        Сохраняет модель многогранника в формате Wavefront OBJ.
        :param filename: имя файла для сохранения
        """
        with open(filename, 'w') as f:
            # Записываем вершины
            for vertex in self.vertices:
                f.write(f"v {vertex.x:.6f} {vertex.y:.6f} {vertex.z:.6f}\n")
            # Записываем грани (индексы начинаются с 1)
            for face in self.faces:
                indices = " ".join(str(idx + 1) for idx in face.vertex_indices)
                f.write(f"f {indices}\n")
    
    @classmethod
    def from_obj(cls, filename: str, name: str = "Loaded Model"):
        """
        Загружает модель многогранника из файла в формате Wavefront OBJ.
        :param filename: имя файла для загрузки
        :param name: название загруженной модели
        :return: экземпляр Polyhedron
        """
        vertices = []
        faces = []
        obj_normals: List[np.ndarray] = []
        normals_per_vertex: dict[int, List[np.ndarray]] = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()[1:]
                    if len(parts) == 3:
                        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                elif line.startswith('vn '):
                    parts = line.split()[1:]
                    if len(parts) == 3:
                        n = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
                        length = np.linalg.norm(n)
                        obj_normals.append(n / length if length > 1e-8 else np.array([0.0, 0.0, 1.0]))
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    face_indices = []
                    for part in parts:
                        # Берем только индекс вершины (игнорируем / для текстур/нормалей)
                        tokens = part.split('/')
                        v_idx = int(tokens[0]) - 1
                        face_indices.append(v_idx)
                        if len(tokens) >= 3 and tokens[2] != '':
                            vn_idx = int(tokens[2]) - 1
                            if 0 <= vn_idx < len(obj_normals):
                                normals_per_vertex.setdefault(v_idx, []).append(obj_normals[vn_idx])
                    faces.append(tuple(face_indices))
        instance = cls(vertices, faces, name)
        if normals_per_vertex:
            averaged_normals: List[np.ndarray] = []
            for vid in range(len(vertices)):
                values = normals_per_vertex.get(vid)
                if not values:
                    averaged_normals.append(np.array([0.0, 0.0, 1.0]))
                else:
                    n = np.mean(values, axis=0)
                    length = np.linalg.norm(n)
                    averaged_normals.append(n / length if length > 1e-8 else np.array([0.0, 0.0, 1.0]))
            instance.obj_vertex_normals = averaged_normals
        # Центрируем модель после загрузки
        center = instance.get_center()
        trans_matrix = translation_matrix(-center.x, -center.y, -center.z)
        instance.apply_transform(trans_matrix)
        return instance
    
    def __repr__(self):
        return f"Polyhedron(name={self.name}, vertices={len(self.vertices)}, faces={len(self.faces)})"

    # --- Освещение ---
    def compute_vertex_normals(self, vertices: List[Point3D]) -> List[np.ndarray]:
        """Усредняет нормали смежных треугольников для каждой вершины."""
        normals = [np.zeros(3, dtype=float) for _ in vertices]
        for face in self.faces:
            if len(face.vertex_indices) < 3:
                continue
            base = vertices[face.vertex_indices[0]]
            for i in range(1, len(face.vertex_indices) - 1):
                v2 = vertices[face.vertex_indices[i]]
                v3 = vertices[face.vertex_indices[i + 1]]
                e1 = np.array([v2.x - base.x, v2.y - base.y, v2.z - base.z])
                e2 = np.array([v3.x - base.x, v3.y - base.y, v3.z - base.z])
                n = np.cross(e1, e2)
                length = np.linalg.norm(n)
                if length < 1e-8:
                    continue
                n = n / length
                for idx in (face.vertex_indices[0], face.vertex_indices[i], face.vertex_indices[i + 1]):
                    normals[idx] += n
        for idx, n in enumerate(normals):
            length = np.linalg.norm(n)
            if length < 1e-8:
                normals[idx] = np.array([0.0, 0.0, 1.0])
            else:
                normals[idx] = n / length
        return normals

    def transform_normals(self, normals: List[np.ndarray], matrix: np.ndarray) -> List[np.ndarray]:
        """Преобразует нормали матрицей (используется инверсия/транспонирование)."""
        try:
            normal_matrix = np.linalg.inv(matrix).T
        except np.linalg.LinAlgError:
            normal_matrix = None
        transformed = []
        for n in normals:
            vec = np.array([n[0], n[1], n[2], 0.0])
            if normal_matrix is None:
                out = vec[:3]
            else:
                out = (normal_matrix @ vec)[:3]
            length = np.linalg.norm(out)
            transformed.append(out / length if length > 1e-8 else np.array([0.0, 0.0, 1.0]))
        return transformed

    def vertex_color_lambert(self, position_view: Point3D, normal_view: np.ndarray, light_view: np.ndarray | None) -> Tuple[int, int, int]:
        """Диффузный цвет по модели Ламберта (используется в шейдинге Гуро)."""
        n = -normal_view  # смотрим в сторону камеры
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-6 else np.array([0.0, 0.0, 1.0])
        if light_view is None:
            ldir = np.array([0.0, 0.0, 1.0])
        else:
            p = np.array([position_view.x, position_view.y, position_view.z, 1.0])
            L = light_view - p
            ldir = L[:3]
            ll = np.linalg.norm(ldir)
            ldir = ldir / ll if ll > 1e-6 else np.array([0.0, 0.0, 1.0])
        diff = max(0.0, float(np.dot(n, ldir)))
        intensity = self.ambient_intensity + (1 - self.ambient_intensity) * diff
        base = np.array(self.object_color) / 255.0
        light = np.array(self.light_color) / 255.0
        rgb = np.clip(base * light * intensity, 0.0, 1.0)
        return tuple((rgb * 255).astype(int))

    def phong_toon_color(self, position_view: Point3D, normal_view: np.ndarray, light_view: np.ndarray | None) -> Tuple[int, int, int]:
        """Toon-шейдинг: квантуем диффузную компоненту модели Фонга."""
        n = -normal_view
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-6 else np.array([0.0, 0.0, 1.0])
        if light_view is None:
            ldir = np.array([0.0, 0.0, 1.0])
        else:
            p = np.array([position_view.x, position_view.y, position_view.z, 1.0])
            L = light_view - p
            ldir = L[:3]
            ll = np.linalg.norm(ldir)
            ldir = ldir / ll if ll > 1e-6 else np.array([0.0, 0.0, 1.0])
        diff = max(0.0, float(np.dot(n, ldir)))
        # Квантование для toon-эффекта
        if diff > 0.9:
            shade = 1.0
        elif diff > 0.6:
            shade = 0.75
        elif diff > 0.35:
            shade = 0.5
        elif diff > 0.15:
            shade = 0.3
        else:
            shade = 0.15
        intensity = max(self.ambient_intensity, shade)
        base = np.array(self.object_color) / 255.0
        light = np.array(self.light_color) / 255.0
        rgb = np.clip(base * light * intensity, 0.0, 1.0)
        return tuple((rgb * 255).astype(int))


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

def create_surface_of_revolution(generatrix: List[Tuple[float, float, float]], axis: str, divisions: int, name="Фигура вращения"):
    """
    Создает фигуру вращения на основе образующей.
    generatrix: список точек (x,y,z)
    axis: 'x', 'y' или 'z' — ось вращения
    divisions: количество разбиений (например 36)
    """
    angle_step = 360.0 / divisions

    vertices = []
    for k in range(divisions):
        angle = math.radians(k * angle_step)
        if axis == 'x':
            R = rotation_x_matrix(math.degrees(angle))
        elif axis == 'y':
            R = rotation_y_matrix(math.degrees(angle))
        elif axis == 'z':
            R = rotation_z_matrix(math.degrees(angle))
        else:
            raise ValueError("Ось должна быть 'x', 'y' или 'z'.")

        for (x, y, z) in generatrix:
            v = np.array([x, y, z, 1.0]) @ R.T
            vertices.append((v[0], v[1], v[2]))

    faces = []
    gcount = len(generatrix)
    for k in range(divisions):
        for i in range(gcount - 1):
            a = k * gcount + i
            b = ((k+1) % divisions) * gcount + i
            c = ((k+1) % divisions) * gcount + (i+1)
            d = k * gcount + (i+1)
            faces.append((a, b, c, d))

    return Polyhedron(vertices, faces, name)

# --- Функция для построения графика двух переменных ---

def create_surface_plot(func, x_range, y_range, steps, name="График функции"):
    """
    Создает сегмент поверхности по функции z = f(x, y).
    :param func: функция двух переменных f(x, y)
    :param x_range: кортеж (x_min, x_max)
    :param y_range: кортеж (y_min, y_max)
    :param steps: количество разбиений по каждой оси
    :param name: название модели
    :return: экземпляр Polyhedron
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_step = (x_max - x_min) / steps
    y_step = (y_max - y_min) / steps
    
    vertices = []
    # Создаем сетку вершин
    for i in range(steps + 1):
        for j in range(steps + 1):
            x = x_min + i * x_step
            y = y_min + j * y_step
            try:
                z = func(x, y)
                if not np.isfinite(z):  # Проверка на inf, -inf, nan
                    z = 0 # Замена на 0, если результат не конечен
            except (ValueError, ZeroDivisionError):
                z = 0 # Обработка математических ошибок
            vertices.append((x, y, z))
            
    faces = []
    # Создаем грани (четырехугольники)
    for i in range(steps):
        for j in range(steps):
            # Индексы вершин для текущего квадрата в сетке
            idx1 = i * (steps + 1) + j
            idx2 = (i + 1) * (steps + 1) + j
            idx3 = (i + 1) * (steps + 1) + (j + 1)
            idx4 = i * (steps + 1) + (j + 1)
            faces.append((idx1, idx2, idx3, idx4))
            
    # Масштабируем для лучшей визуализации
    poly = Polyhedron(vertices, faces, name)
    center = poly.get_center()
    poly.apply_transform(translation_matrix(-center.x, -center.y, -center.z))
    
    # Найдем максимальный размер и смасштабируем до ~200
    max_dim = max(abs(v.x) for v in poly.vertices)
    max_dim = max(max_dim, max(abs(v.y) for v in poly.vertices))
    max_dim = max(max_dim, max(abs(v.z) for v in poly.vertices))
    if max_dim > 1e-6:
        scale_factor = 150 / max_dim
        poly.apply_transform(scale_matrix(scale_factor, scale_factor, scale_factor))
        
    return poly


# --- Z-buffer функции ---

def barycentric_coords(p: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Вычисляет барицентрические координаты точки p относительно треугольника (a, b, c).
    Возвращает (alpha, beta, gamma), или None если треугольник вырожден.
    """
    det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(det) < 1e-6:
        return None  # Вырожденный треугольник

    alpha = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
    beta = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det
    gamma = 1.0 - alpha - beta
    
    return alpha, beta, gamma

def interpolate_z(coords: Tuple[float, float, float], z_a: float, z_b: float, z_c: float) -> float:
    """Интерполирует Z-координату с использованием барицентрических координат."""
    alpha, beta, gamma = coords
    return alpha * z_a + beta * z_b + gamma * z_c

# --- Класс КАМЕРЫ (Новая функция) ---

class Camera:
    """
    Класс камеры для вращения вокруг статического объекта.
    Управляется сферическими координатами (theta, phi) и дистанцией.
    """
    def __init__(self, distance: float, target: Point3D = Point3D(0, 0, 0)):
        self.distance = distance
        self.target = np.array([target.x, target.y, target.z])
        # Начальные углы:
        self.theta = -math.pi / 2  # Азимут
        self.phi = math.pi / 2     # Зенит
        self.up = np.array([0, 1, 0])

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Возвращает матрицу ВРАЩЕНИЯ камеры.
        Важно: Мы не включаем перенос (Translation), так как функция project_perspective
        уже использует camera_distance в делителе (фактически перемещая камеру по Z).
        Поэтому здесь нам нужно только повернуть мир так, чтобы он "смотрел" на камеру.
        """
        # 1. Позиция камеры в сферических координатах
        x = self.distance * math.sin(self.phi) * math.cos(self.theta)
        y = self.distance * math.cos(self.phi)
        z = self.distance * math.sin(self.phi) * math.sin(self.theta)
        
        position = np.array([x, y, z])
        
        # 2. Вектора базиса (LookAt)
        # Forward (F): от камеры к цели
        f = self.target - position
        f = f / np.linalg.norm(f)
        
        # Right (R): перпендикулярно F и Up
        r = np.cross(f, self.up)
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1, 0, 0])
        r = r / np.linalg.norm(r)
        
        # True Up (U): перпендикулярно R и F
        u = np.cross(r, f)
        
        # 3. Матрица вращения (строки - это базисные вектора камеры)
        M = np.eye(4)
        M[0, :3] = r
        M[1, :3] = -u  # Инвертируем Y для экранных координат (Y вниз)
        M[2, :3] = f   # Z смотрит от камеры
        
        return M

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
    
    # Инициализация камеры
    camera_obj = Camera(camera_distance)
    use_orbit_camera = False # Флаг использования орбитальной камеры

    # Матрица поворота камеры (для изменения вектора обзора)
    camera_rotation = np.eye(4)
    
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
        '0': lambda: create_surface_plot(
            func=lambda x, y: 50 * np.sin(np.sqrt(x**2 + y**2) / 10),
            x_range=(-50, 50),
            y_range=(-50, 50),
            steps=30
        ),
        '9': lambda: create_surface_of_revolution(
        generatrix=[(0, -100, 0), (40, -80, 0), (60, -20, 0), (40, 60, 0), (0, 100, 0)],
        axis='z',
        divisions=48,
        name="Фигура вращения"
    )
    }
    
    current_poly_key = '2'
    polyhedron = polyhedrons[current_poly_key]()
    current_shading_mode = 'flat'
    polyhedron.shading_mode = current_shading_mode
    
    auto_rotate = {'x': False, 'y': True, 'z': False}
    projection_mode = 'perspective'  # 'perspective' или 'axonometric'
    
    # Имя файла для загрузки/сохранения OBJ (можно изменить)
    obj_filename = "model.obj"
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Смена фигур
                if '0' <= event.unicode <= '5':
                    current_poly_key = event.unicode
                    polyhedron = polyhedrons[current_poly_key]()
                    polyhedron.shading_mode = current_shading_mode

                if '9' == event.unicode:
                    current_poly_key = event.unicode
                    polyhedron = polyhedrons[current_poly_key]()    
                    polyhedron.shading_mode = current_shading_mode
                
                # Сброс
                if event.key == pygame.K_r:
                    polyhedron = polyhedrons[current_poly_key]()
                    polyhedron.shading_mode = current_shading_mode
                
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

                # Режимы шейдинга
                if event.key == pygame.K_F1:
                    current_shading_mode = 'flat'
                    polyhedron.shading_mode = current_shading_mode
                if event.key == pygame.K_F2:
                    current_shading_mode = 'gouraud'
                    polyhedron.shading_mode = current_shading_mode
                if event.key == pygame.K_F3:
                    current_shading_mode = 'phong_toon'
                    polyhedron.shading_mode = current_shading_mode

                # Поворот вокруг произвольной прямой (по умолчанию — arbitrary_p1->arbitrary_p2)
                if event.key == pygame.K_k:
                    M = rotation_about_line(arbitrary_p1, arbitrary_p2, arbitrary_angle_step)
                    polyhedron.apply_transform(M)

                # Повороты вокруг прямой, проходящей через центр, параллельно выбранной оси
                if event.key == pygame.K_u:  # вокруг оси X через центр
                    center = polyhedron.get_center()
                    M = rotation_axis_through_center_matrix(center, 'x', 10)
                    polyhedron.apply_transform(M)
                if event.key == pygame.K_i and not use_orbit_camera:  # Исключаем конфликт с I если не активна орбита
                    center = polyhedron.get_center()
                    M = rotation_axis_through_center_matrix(center, 'y', 10)
                    polyhedron.apply_transform(M)
                if event.key == pygame.K_o:  # вокруг оси Z через центр
                    center = polyhedron.get_center()
                    M = rotation_axis_through_center_matrix(center, 'z', 10)
                    polyhedron.apply_transform(M)

                # Сохранение в OBJ
                if event.key == pygame.K_f:  # F для "file save"
                    polyhedron.to_obj(obj_filename)
                    print(f"Модель сохранена в {obj_filename}")

                # Загрузка из OBJ
                if event.key == pygame.K_g:  # G для "get load"
                    try:
                        polyhedron = Polyhedron.from_obj(obj_filename)
                        print(f"Модель загружена из {obj_filename}")
                    except FileNotFoundError:
                        print(f"Файл {obj_filename} не найден")
                    except Exception as e:
                        print(f"Ошибка загрузки: {e}")

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
        
        # Поворот вручную (поворот объекта)
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

        # Авто-вращение (вращение объекта)
        if auto_rotate['x']: 
            polyhedron.apply_transform(rotation_x_matrix(rotation_speed / 2))
        if auto_rotate['y']: 
            polyhedron.apply_transform(rotation_y_matrix(rotation_speed / 2))
        if auto_rotate['z']: 
            polyhedron.apply_transform(rotation_z_matrix(rotation_speed / 2))
        
        # --- УПРАВЛЕНИЕ КАМЕРОЙ (НОВОЕ ЗАДАНИЕ) ---
        
        camera_changed = False
        if keys[pygame.K_c]:
            camera_obj.theta -= 0.03
            use_orbit_camera = True
            camera_changed = True
        if keys[pygame.K_v]:
            camera_obj.theta += 0.03
            use_orbit_camera = True
            camera_changed = True
        if keys[pygame.K_b]:
            camera_obj.phi = max(0.1, camera_obj.phi - 0.03) # Ограничение, чтобы не перевернулось
            use_orbit_camera = True
            camera_changed = True
        if keys[pygame.K_n]:
            camera_obj.phi = min(math.pi - 0.1, camera_obj.phi + 0.03)
            use_orbit_camera = True
            camera_changed = True
        if keys[pygame.K_m]: # Сброс по заданию
            camera_obj.theta = -math.pi / 2
            camera_obj.phi = math.pi / 2
            use_orbit_camera = True
            camera_changed = True
        
        if use_orbit_camera:
            camera_rotation = camera_obj.get_rotation_matrix()
        else:
            # Старое управление (если не трогали орбитальную камеру)
            # Изменение вектора обзора (поворот камеры)
            if keys[pygame.K_j]:
                camera_rotation = rotation_y_matrix(-rotation_speed) @ camera_rotation
            if keys[pygame.K_l]:
                camera_rotation = rotation_y_matrix(rotation_speed) @ camera_rotation
            if keys[pygame.K_i]:
                camera_rotation = rotation_x_matrix(-rotation_speed) @ camera_rotation
            # M был занят старой функцией, но теперь он на Reset камеры.
            # Для совместимости можно оставить тут логику, если M не нажат как ресет
            pass 

        # Отрисовка
        screen.fill(polyhedron.bg_color)
        z_buffer = np.full((screen_height, screen_width), np.inf, dtype=float)
        polyhedron.draw(screen, camera_distance, screen_width, screen_height, projection_mode, camera_rotation, z_buffer)
        
        # Интерфейс
        info = [
            polyhedron.get_info(),
            f"Проекция: {projection_mode}",
            f"Шейдинг: {current_shading_mode} (F1/F2/F3)",
            "",
            "Управление:",
            "1-5: Сменить фигуру",
            "0: Построение графика функции",
            "9: Построение фигуры вращения",
            "Стрелки: Смещение",
            "W/S, A/D, Q/E: Поворот объекта",
            "+/-: Масштаб (относительно центра)",
            "R: Сброс",
            "X/Y/Z: Авто-вращение",
            "6/7/8: Отражения относительно XY/XZ/YZ",
            "P: Переключить перспективу/аксонометрию",
            "F1/F2/F3: Flat / Gouraud / Phong-Toon",
            "U/I/O: Поворот вокруг прямой через центр (X/Y/Z) на 10°",
            "K: Поворот вокруг произвольной прямой (по умолчанию задается в коде)",
            f"Arb line p1={arbitrary_p1} p2={arbitrary_p2} step={arbitrary_angle_step}° (нажмите K)",
            "F: Сохранить в OBJ (model.obj)",
            "G: Загрузить из OBJ (model.obj)",
            "--- НОВОЕ УПРАВЛЕНИЕ КАМЕРОЙ ---",
            "C / V: Вращение влево / вправо (Азимут)",
            "B / N: Вращение вверх / вниз (Зенит)",
            "M: Сброс камеры"
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
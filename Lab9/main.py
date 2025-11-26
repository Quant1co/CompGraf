import pygame
import numpy as np
import math
from typing import List, Tuple

# --- Классы ---

class Point3D:
    """
    Класс для представления точки в 3D пространстве.
    Добавлены u, v - текстурные координаты.
    """
    def __init__(self, x: float, y: float, z: float, u: float = 0.0, v: float = 0.0):
        """
        Инициализация точки в 3D пространстве.
        """
        self.x = x
        self.y = y
        self.z = z
        self.u = u
        self.v = v
        # Храним точку как вектор в однородных координатах для матричных преобразований
        self.homogeneous = np.array([x, y, z, 1], dtype=float)
    
    def apply_transform(self, matrix: np.ndarray):
        """
        Применяет матрицу преобразования к точке.
        """
        self.homogeneous = self.homogeneous @ matrix.T
        self.x, self.y, self.z = self.homogeneous[:3]
    
    def project_perspective(self, camera_distance: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """
        Перспективная проекция.
        """
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
        return Point3D(self.x, self.y, self.z, self.u, self.v)
    
    def __repr__(self):
        return f"Point3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f}, u={self.u:.2f}, v={self.v:.2f})"


class Polygon:
    """
    Класс для представления многоугольника (грани).
    """
    def __init__(self, vertex_indices: List[int], color: Tuple[int, int, int] = None):
        self.vertex_indices = vertex_indices
        self.color = color if color else (200, 200, 255)
        self.fill_color = None  # Цвет заливки (если нужна заливка)
    
    def get_vertices(self, all_vertices: List[Point3D]) -> List[Point3D]:
        return [all_vertices[i] for i in self.vertex_indices]
    
    def calculate_normal(self, vertices: List[Point3D]) -> np.ndarray:
        if len(self.vertex_indices) < 3:
            return np.array([0, 0, 1])
        
        p1 = vertices[self.vertex_indices[0]]
        p2 = vertices[self.vertex_indices[1]]
        p3 = vertices[self.vertex_indices[2]]
        
        v1 = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        v2 = np.array([p3.x - p1.x, p3.y - p1.y, p3.z - p1.z])
        
        normal = np.cross(v1, v2)
        return normal
    
    def draw(self, surface, projected_points: List[Tuple[int, int]], line_width: int = 1):
        points = [projected_points[i] for i in self.vertex_indices]
        pygame.draw.polygon(surface, self.color, points, line_width)
    
    def __repr__(self):
        return f"Polygon(vertices={self.vertex_indices})"


class Polyhedron:
    """
    Класс для представления многогранника.
    """
    def __init__(self, vertices: List[Point3D], 
                 faces: List[Tuple[int, ...]], name: str = "Polyhedron"):
        self.name = name
        
        # Обработка входных данных: если переданы кортежи (старый способ), конвертируем в Point3D
        if vertices and isinstance(vertices[0], tuple):
             self.vertices = [Point3D(x, y, z) for x, y, z in vertices]
        else:
            self.vertices = vertices

        self.faces = [Polygon(list(face)) for face in faces]
        self.normalize_face_orientations()
        self.edge_color = (200, 200, 255)
        self.bg_color = (10, 20, 40)
        self.show_faces = False
        
        # Параметры освещения/шейдинга
        # ЗАДАНИЕ 1: Добавить положение источника света и цвет объекта
        self.object_color = (180, 200, 255)
        self.light_color = (255, 255, 255)
        self.light_position = np.array([400.0, 400.0, 600.0, 1.0])
        self.ambient_intensity = 0.2
        self.shading_mode = 'flat'
        
        # Текстура
        self.texture: pygame.Surface | None = None 
        
        self.obj_vertex_normals: List[np.ndarray] | None = None
    
    def normalize_face_orientations(self):
        center = self.get_center()
        center_vec = np.array([center.x, center.y, center.z])
        
        for face in self.faces:
            normal = face.calculate_normal(self.vertices)
            if np.linalg.norm(normal) == 0:
                continue
            
            p = np.array([self.vertices[face.vertex_indices[0]].x, 
                          self.vertices[face.vertex_indices[0]].y, 
                          self.vertices[face.vertex_indices[0]].z])
            vector_to_p = p - center_vec
            if np.dot(normal, vector_to_p) < 0:
                face.vertex_indices.reverse()
                normal = face.calculate_normal(self.vertices)
    
    def apply_transform(self, matrix: np.ndarray):
        """
        ЗАДАНИЕ 1: Добавить возможность применения аффинных преобразований к объекту.
        """
        for vertex in self.vertices:
            vertex.apply_transform(matrix)
        if self.obj_vertex_normals is not None:
            # Для нормалей используем транспонированную обратную матрицу,
            # чтобы сохранить перпендикулярность поверхности при масштабировании.
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
        self.normalize_face_orientations()
    
    def get_center(self) -> Point3D:
        if not self.vertices:
            return Point3D(0, 0, 0)
        avg_x = sum(v.x for v in self.vertices) / len(self.vertices)
        avg_y = sum(v.y for v in self.vertices) / len(self.vertices)
        avg_z = sum(v.z for v in self.vertices) / len(self.vertices)
        return Point3D(avg_x, avg_y, avg_z)
    
    def draw(self, surface, camera_distance: float, screen_width: int, screen_height: int, projection_mode: str = 'perspective', camera_rotation: np.ndarray = np.eye(4), z_buffer: np.ndarray = None):
        """
        Проецирует и отрисовывает многогранник с использованием Z-буфера и текстурирования.
        """
        if z_buffer is None:
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
            # Копируем UV координаты при создании точки в пространстве камеры
            viewed = Point3D(v_hom[0], v_hom[1], v_hom[2], vertex.u, vertex.v)
            viewed_vertices.append(viewed)
        
        # Проецируем вершины
        if projection_mode == 'perspective':
            projected_points = [v.project_perspective(camera_distance, screen_width, screen_height) for v in viewed_vertices]
        else:
            projected_points = [v.project_axonometric(screen_width, screen_height) for v in viewed_vertices]

        light_view = None
        if self.light_position is not None:
            light_view = self.light_position @ full_view.T

        if self.obj_vertex_normals is not None:
            vertex_normals_view = self.transform_normals(self.obj_vertex_normals, full_view)
        else:
            vertex_normals_view = self.compute_vertex_normals(viewed_vertices)
        
        # Подготовка текстуры
        tex_width, tex_height = 0, 0
        if self.texture:
            tex_width = self.texture.get_width()
            tex_height = self.texture.get_height()

        # Отрисовываем грани
        for face in self.faces:
            # Backface culling
            # ЗАДАНИЕ 1: Удаление нелицевых граней (если нормаль смотрит от камеры)
            normal = face.calculate_normal(viewed_vertices)
            if normal[2] > 1e-6:
                continue

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

            # Разбиваем на треугольники
            v0_idx = face.vertex_indices[0]
            for i in range(1, len(face.vertex_indices) - 1):
                v1_idx = face.vertex_indices[i]
                v2_idx = face.vertex_indices[i + 1]
                
                p0, p1, p2 = projected_points[v0_idx], projected_points[v1_idx], projected_points[v2_idx]
                z0, z1, z2 = viewed_vertices[v0_idx].z, viewed_vertices[v1_idx].z, viewed_vertices[v2_idx].z
                
                # Получаем UV координаты вершин
                u0, u1, u2 = viewed_vertices[v0_idx].u, viewed_vertices[v1_idx].u, viewed_vertices[v2_idx].u
                v0, v1, v2 = viewed_vertices[v0_idx].v, viewed_vertices[v1_idx].v, viewed_vertices[v2_idx].v

                if self.shading_mode == 'gouraud':
                    # ЗАДАНИЕ 1.1: Шейдинг Гуро. Вычисляем цвет в каждой вершине по модели Ламберта.
                    c0 = np.array(self.vertex_color_lambert(viewed_vertices[v0_idx], vertex_normals_view[v0_idx], light_view))
                    c1 = np.array(self.vertex_color_lambert(viewed_vertices[v1_idx], vertex_normals_view[v1_idx], light_view))
                    c2 = np.array(self.vertex_color_lambert(viewed_vertices[v2_idx], vertex_normals_view[v2_idx], light_view))
                elif self.shading_mode == 'phong_toon':
                    # ЗАДАНИЕ 1.2: Шейдинг Фонга. Подготовка нормалей для интерполяции.
                    n0 = vertex_normals_view[v0_idx]
                    n1 = vertex_normals_view[v1_idx]
                    n2 = vertex_normals_view[v2_idx]
                    pv0 = np.array([viewed_vertices[v0_idx].x, viewed_vertices[v0_idx].y, viewed_vertices[v0_idx].z])
                    pv1 = np.array([viewed_vertices[v1_idx].x, viewed_vertices[v1_idx].y, viewed_vertices[v1_idx].z])
                    pv2 = np.array([viewed_vertices[v2_idx].x, viewed_vertices[v2_idx].y, viewed_vertices[v2_idx].z])

                x_min = max(0, min(p0[0], p1[0], p2[0]))
                x_max = min(screen_width - 1, max(p0[0], p1[0], p2[0]))
                y_min = max(0, min(p0[1], p1[1], p2[1]))
                y_max = min(screen_height - 1, max(p0[1], p1[1], p2[1]))
                
                if (x_max - x_min) * (y_max - y_min) > 500000: 
                    continue

                # Растеризация
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        coords = barycentric_coords((x, y), p0, p1, p2)
                        if coords is None: continue
                        
                        alpha, beta, gamma = coords
                        if alpha >= 0 and beta >= 0 and gamma >= 0:
                            interpolated_z = interpolate_z(coords, z0, z1, z2)
                            
                            if interpolated_z < z_buffer[y, x]:
                                z_buffer[y, x] = interpolated_z
                                
                                # -- ЛОГИКА ТЕКСТУРИРОВАНИЯ --
                                if self.texture:
                                    # Интерполируем UV
                                    u = alpha * u0 + beta * u1 + gamma * u2
                                    v = alpha * v0 + beta * v1 + gamma * v2
                                    
                                    # Перевод в координаты пикселя текстуры
                                    tex_x = int(u * (tex_width - 1)) % tex_width
                                    tex_y = int(v * (tex_height - 1)) % tex_height
                                    
                                    # Берем цвет
                                    tex_color = self.texture.get_at((tex_x, tex_y))[:3]
                                    
                                    # Для простоты смешиваем текстуру с Flat shading (интенсивность)
                                    if self.shading_mode == 'flat':
                                        # Грубая оценка интенсивности по flat цвету
                                        intensity = max(face_color) / 255.0
                                        final_color = tuple([min(255, int(c * intensity)) for c in tex_color])
                                        surface.set_at((x, y), final_color)
                                    else:
                                        # В других режимах просто рисуем текстуру (можно доработать для Phong/Gouraud)
                                        surface.set_at((x, y), tex_color)
                                else:
                                    # Обычная отрисовка без текстур
                                    if self.shading_mode == 'flat':
                                        surface.set_at((x, y), face_color)
                                    elif self.shading_mode == 'gouraud':
                                        # ЗАДАНИЕ 1.1: Интерполируем цвет между цветами вершин (билинейная интерполяция).
                                        col = alpha * c0 + beta * c1 + gamma * c2
                                        col = np.clip(col, 0, 255).astype(int)
                                        surface.set_at((x, y), tuple(col))
                                    elif self.shading_mode == 'phong_toon':
                                        # ЗАДАНИЕ 1.2: Шейдинг Фонга.
                                        # 1. Интерполируем нормали между вершинами.
                                        n_interp = alpha * n0 + beta * n1 + gamma * n2
                                        # 2. Нормализация интерполированной нормали.
                                        n_len = np.linalg.norm(n_interp)
                                        n_hat = n_interp / n_len if n_len > 1e-6 else np.array([0.0, 0.0, 1.0])
                                        
                                        pos_interp = alpha * pv0 + beta * pv1 + gamma * pv2
                                        pos_point = Point3D(pos_interp[0], pos_interp[1], pos_interp[2])
                                        
                                        # 3. Вычисляем цвет пикселя (модель туншейдинга).
                                        color = self.phong_toon_color(pos_point, n_hat, light_view)
                                        surface.set_at((x, y), color)
            
            # Рисуем ребра только если текстура выключена (иначе портит вид)
            if not self.texture:
                indices = face.vertex_indices
                for i in range(len(indices)):
                    idx_a = indices[i]
                    idx_b = indices[(i + 1) % len(indices)]
                    draw_line_with_z(
                        surface, 
                        z_buffer, 
                        projected_points[idx_a], 
                        projected_points[idx_b], 
                        viewed_vertices[idx_a].z, 
                        viewed_vertices[idx_b].z, 
                        self.edge_color
                    )

    def draw_legacy(self, surface, camera_distance: float, screen_width: int, screen_height: int, projection_mode: str = 'perspective', camera_rotation: np.ndarray = np.eye(4)):
        """
        Проецирует и отрисовывает многогранник (старый метод с сортировкой).
        """
        if projection_mode == 'axonometric':
            proj_matrix = axonometric_view_matrix()
        else:
            proj_matrix = np.eye(4)
        
        full_view = proj_matrix @ camera_rotation
        
        viewed_vertices = []
        for vertex in self.vertices:
            v_hom = vertex.homogeneous @ full_view.T
            viewed = Point3D(v_hom[0], v_hom[1], v_hom[2])
            viewed_vertices.append(viewed)
        
        if projection_mode == 'perspective':
            projected_points = [v.project_perspective(camera_distance, screen_width, screen_height) for v in viewed_vertices]
        else:  # axonometric
            projected_points = [v.project_axonometric(screen_width, screen_height) for v in viewed_vertices]
        
        face_depths = []
        for face in self.faces:
            avg_z = sum(viewed_vertices[i].z for i in face.vertex_indices) / len(face.vertex_indices)
            face_depths.append((face, avg_z))
        
        face_depths.sort(key=lambda x: x[1], reverse=True)
        
        for face, _ in face_depths:
            normal = face.calculate_normal(viewed_vertices)
            if normal[2] < -1e-6:
                points = [projected_points[i] for i in face.vertex_indices]
                pygame.draw.polygon(surface, (50, 50, 80), points)
                face.draw(surface, projected_points)
    
    def get_info(self) -> str:
        return f"{self.name}: {len(self.vertices)} вершин, {len(self.faces)} граней"
    
    def to_obj(self, filename: str):
        with open(filename, 'w') as f:
            for vertex in self.vertices:
                f.write(f"v {vertex.x:.6f} {vertex.y:.6f} {vertex.z:.6f}\n")
            for face in self.faces:
                indices = " ".join(str(idx + 1) for idx in face.vertex_indices)
                f.write(f"f {indices}\n")
    
    @classmethod
    def from_obj(cls, filename: str, name: str = "Loaded Model"):
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
                        # При загрузке из обычного OBJ без текстур ставим UV в 0
                        vertices.append(Point3D(float(parts[0]), float(parts[1]), float(parts[2])))
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
        center = instance.get_center()
        trans_matrix = translation_matrix(-center.x, -center.y, -center.z)
        instance.apply_transform(trans_matrix)
        return instance
    
    def __repr__(self):
        return f"Polyhedron(name={self.name}, vertices={len(self.vertices)}, faces={len(self.faces)})"

    # --- Освещение ---
    def compute_vertex_normals(self, vertices: List[Point3D]) -> List[np.ndarray]:
        """
        ЗАДАНИЕ 1: Вычислить нормаль к каждой вершине.
        Нормаль вершины вычисляется как усредненная нормаль прилегающих граней.
        """
        normals = [np.zeros(3, dtype=float) for _ in vertices]
        for face in self.faces:
            if len(face.vertex_indices) < 3: continue
            base = vertices[face.vertex_indices[0]]
            for i in range(1, len(face.vertex_indices) - 1):
                v2 = vertices[face.vertex_indices[i]]
                v3 = vertices[face.vertex_indices[i + 1]]
                e1 = np.array([v2.x - base.x, v2.y - base.y, v2.z - base.z])
                e2 = np.array([v3.x - base.x, v3.y - base.y, v3.z - base.z])
                n = np.cross(e1, e2)
                length = np.linalg.norm(n)
                if length < 1e-8: continue
                n = n / length
                for idx in (face.vertex_indices[0], face.vertex_indices[i], face.vertex_indices[i + 1]):
                    normals[idx] += n
        for idx, n in enumerate(normals):
            length = np.linalg.norm(n)
            if length < 1e-8: normals[idx] = np.array([0.0, 0.0, 1.0])
            else: normals[idx] = n / length
        return normals

    def transform_normals(self, normals: List[np.ndarray], matrix: np.ndarray) -> List[np.ndarray]:
        try:
            normal_matrix = np.linalg.inv(matrix).T
        except np.linalg.LinAlgError:
            normal_matrix = None
        transformed = []
        for n in normals:
            vec = np.array([n[0], n[1], n[2], 0.0])
            out = (normal_matrix @ vec)[:3] if normal_matrix is not None else vec[:3]
            length = np.linalg.norm(out)
            transformed.append(out / length if length > 1e-8 else np.array([0.0, 0.0, 1.0]))
        return transformed

    def vertex_color_lambert(self, position_view: Point3D, normal_view: np.ndarray, light_view: np.ndarray | None) -> Tuple[int, int, int]:
        """
        ЗАДАНИЕ 1.1: Вычислить цвет по модели Ламберта (диффузное отражение).
        I = Ia + Id * (N * L)
        """
        n = -normal_view
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-6 else np.array([0.0, 0.0, 1.0])
        if light_view is None: ldir = np.array([0.0, 0.0, 1.0])
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
        """
        ЗАДАНИЕ 1.2: Вычислить цвет в соответствии с моделью туншейдинга.
        Дискретизация интенсивности освещения.
        """
        n = -normal_view
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-6 else np.array([0.0, 0.0, 1.0])
        if light_view is None: ldir = np.array([0.0, 0.0, 1.0])
        else:
            p = np.array([position_view.x, position_view.y, position_view.z, 1.0])
            L = light_view - p
            ldir = L[:3]
            ll = np.linalg.norm(ldir)
            ldir = ldir / ll if ll > 1e-6 else np.array([0.0, 0.0, 1.0])
        diff = max(0.0, float(np.dot(n, ldir)))
        if diff > 0.9: shade = 1.0
        elif diff > 0.6: shade = 0.75
        elif diff > 0.35: shade = 0.5
        elif diff > 0.15: shade = 0.3
        else: shade = 0.15
        intensity = max(self.ambient_intensity, shade)
        base = np.array(self.object_color) / 255.0
        light = np.array(self.light_color) / 255.0
        rgb = np.clip(base * light * intensity, 0.0, 1.0)
        return tuple((rgb * 255).astype(int))


# --- Матрицы аффинных преобразований (без изменений) ---
def translation_matrix(tx, ty, tz):
    return np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

def scale_matrix(sx, sy, sz):
    return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])

def rotation_x_matrix(angle):
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def rotation_y_matrix(angle):
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def rotation_z_matrix(angle):
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def reflection_matrix(plane: str):
    if plane == 'xy': return scale_matrix(1, 1, -1)
    elif plane == 'xz': return scale_matrix(1, -1, 1)
    elif plane == 'yz': return scale_matrix(-1, 1, 1)
    else: raise ValueError("Неверная плоскость для отражения")

def rodrigues_rotation_matrix(u: np.ndarray, angle_rad: float) -> np.ndarray:
    ux, uy, uz = u
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    I = np.eye(3)
    uu = np.outer(u, u)
    u_x = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    R = c * I + (1 - c) * uu + s * u_x
    return R

def rotation_about_line(p1: Tuple[float, float, float], p2: Tuple[float, float, float], angle_deg: float) -> np.ndarray:
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    u = p2 - p1
    norm = np.linalg.norm(u)
    if norm == 0: raise ValueError("Две точки совпадают")
    u = u / norm
    angle_rad = math.radians(angle_deg)
    R3 = rodrigues_rotation_matrix(u, angle_rad)
    R = np.eye(4)
    R[:3, :3] = R3
    T1 = translation_matrix(-p1[0], -p1[1], -p1[2])
    T2 = translation_matrix(p1[0], p1[1], p1[2])
    return T2 @ R @ T1

def rotation_axis_through_center_matrix(center: Point3D, axis: str, angle_deg: float) -> np.ndarray:
    if axis == 'x': R = rotation_x_matrix(angle_deg)
    elif axis == 'y': R = rotation_y_matrix(angle_deg)
    elif axis == 'z': R = rotation_z_matrix(angle_deg)
    else: raise ValueError("Неверная ось")
    trans_to_origin = translation_matrix(-center.x, -center.y, -center.z)
    trans_back = translation_matrix(center.x, center.y, center.z)
    return trans_back @ R @ trans_to_origin

def axonometric_view_matrix():
    Rx = rotation_x_matrix(35.26438968)
    Ry = rotation_y_matrix(45)
    return Ry @ Rx

# --- Функции для создания многогранников (С UV координатами) ---

def create_tetrahedron(scale=100):
    """Создает правильный тетраэдр с UV координатами."""
    s = scale / np.sqrt(3)
    coords = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    raw_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    
    vertices = []
    faces = []
    # UV для треугольника
    uvs = [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0)]
    
    for raw_face in raw_faces:
        base_idx = len(vertices)
        current_face_indices = []
        for i, v_idx in enumerate(raw_face):
            x, y, z = coords[v_idx]
            vertices.append(Point3D(x*s, y*s, z*s, uvs[i][0], uvs[i][1]))
            current_face_indices.append(base_idx + i)
        faces.append(tuple(current_face_indices))
    return Polyhedron(vertices, faces, "Тетраэдр")

def create_hexahedron(scale=100):
    """Создает куб с UV координатами."""
    s = scale
    c = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
    ]
    raw_faces = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 7, 4),
        (1, 2, 6, 5), (0, 1, 5, 4), (3, 2, 6, 7)
    ]
    
    vertices = []
    faces = []
    # UV для квадрата
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    
    for raw_face in raw_faces:
        base_idx = len(vertices)
        current_face_indices = []
        for i, v_idx in enumerate(raw_face):
            x, y, z = c[v_idx]
            vertices.append(Point3D(x, y, z, uvs[i][0], uvs[i][1]))
            current_face_indices.append(base_idx + i)
        faces.append(tuple(current_face_indices))
    return Polyhedron(vertices, faces, "Гексаэдр (Куб)")

def create_octahedron(scale=120):
    """Создает правильный октаэдр с UV координатами."""
    s = scale
    coords = [(s, 0, 0), (-s, 0, 0), (0, s, 0), (0, -s, 0), (0, 0, s), (0, 0, -s)]
    raw_faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 2, 5), (1, 5, 3), (1, 3, 4), (1, 4, 2)
    ]
    vertices = []
    faces = []
    uvs = [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for raw_face in raw_faces:
        base_idx = len(vertices)
        current_face_indices = []
        for i, v_idx in enumerate(raw_face):
            x, y, z = coords[v_idx]
            vertices.append(Point3D(x, y, z, uvs[i][0], uvs[i][1]))
            current_face_indices.append(base_idx + i)
        faces.append(tuple(current_face_indices))
    return Polyhedron(vertices, faces, "Октаэдр")

def create_icosahedron(scale=120):
    phi = (1 + math.sqrt(5)) / 2
    s = scale
    vertices_raw = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
    ]
    pts = [Point3D(v[0]*s/phi, v[1]*s/phi, v[2]*s/phi) for v in vertices_raw]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    return Polyhedron(pts, faces, "Икосаэдр")

def create_dodecahedron(scale=80):
    phi = (1 + math.sqrt(5)) / 2
    s = scale
    vertices_raw = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        (0, 1/phi, phi), (0, 1/phi, -phi), (0, -1/phi, phi), (0, -1/phi, -phi),
        (1/phi, phi, 0), (1/phi, -phi, 0), (-1/phi, phi, 0), (-1/phi, -phi, 0),
        (phi, 0, 1/phi), (phi, 0, -1/phi), (-phi, 0, 1/phi), (-phi, 0, -1/phi)
    ]
    pts = [Point3D(v[0]*s, v[1]*s, v[2]*s) for v in vertices_raw]
    faces = [
        (0, 16, 2, 10, 8), (0, 8, 4, 14, 12), (0, 12, 1, 17, 16),
        (1, 9, 5, 14, 12), (1, 17, 3, 11, 9), (2, 16, 17, 3, 13),
        (2, 13, 15, 6, 10), (3, 11, 7, 15, 13), (4, 8, 10, 6, 18),
        (4, 18, 19, 5, 14), (5, 19, 7, 11, 9), (6, 15, 7, 19, 18)
    ]
    return Polyhedron(pts, faces, "Додекаэдр")

def create_surface_of_revolution(generatrix: List[Tuple[float, float, float]], axis: str, divisions: int, name="Фигура вращения"):
    angle_step = 360.0 / divisions
    vertices = []
    for k in range(divisions):
        angle = math.radians(k * angle_step)
        if axis == 'x': R = rotation_x_matrix(math.degrees(angle))
        elif axis == 'y': R = rotation_y_matrix(math.degrees(angle))
        elif axis == 'z': R = rotation_z_matrix(math.degrees(angle))
        else: raise ValueError
        for (x, y, z) in generatrix:
            v = np.array([x, y, z, 1.0]) @ R.T
            vertices.append(Point3D(v[0], v[1], v[2]))

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

def create_surface_plot(func, x_range, y_range, steps, name="График функции"):
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_step = (x_max - x_min) / steps
    y_step = (y_max - y_min) / steps
    vertices = []
    for i in range(steps + 1):
        for j in range(steps + 1):
            x = x_min + i * x_step
            y = y_min + j * y_step
            try: z = func(x, y)
            except: z = 0
            if not np.isfinite(z): z = 0
            u, v = i/steps, j/steps
            vertices.append(Point3D(x, y, z, u, v))
    faces = []
    for i in range(steps):
        for j in range(steps):
            idx1 = i * (steps + 1) + j
            idx2 = (i + 1) * (steps + 1) + j
            idx3 = (i + 1) * (steps + 1) + (j + 1)
            idx4 = i * (steps + 1) + (j + 1)
            faces.append((idx1, idx2, idx3, idx4))
    poly = Polyhedron(vertices, faces, name)
    center = poly.get_center()
    poly.apply_transform(translation_matrix(-center.x, -center.y, -center.z))
    max_dim = max(max(abs(v.x), abs(v.y), abs(v.z)) for v in poly.vertices)
    if max_dim > 1e-6: poly.apply_transform(scale_matrix(150/max_dim, 150/max_dim, 150/max_dim))
    return poly


# --- Z-buffer функции ---
def barycentric_coords(p: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> Tuple[float, float, float]:
    det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(det) < 1e-6: return None
    alpha = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
    beta = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det
    gamma = 1.0 - alpha - beta
    return alpha, beta, gamma

def interpolate_z(coords: Tuple[float, float, float], z_a: float, z_b: float, z_c: float) -> float:
    alpha, beta, gamma = coords
    return alpha * z_a + beta * z_b + gamma * z_c

def draw_line_with_z(surface, z_buffer: np.ndarray, p_start: Tuple[int, int], p_end: Tuple[int, int],
                     z_start: float, z_end: float, color: Tuple[int, int, int]):
    """Рисует линию, учитывая z-буфер, чтобы не показывать скрытые рёбра."""
    x0, y0 = p_start
    x1, y1 = p_end

    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))

    if steps == 0:
        if 0 <= x0 < z_buffer.shape[1] and 0 <= y0 < z_buffer.shape[0]:
            if z_start <= z_buffer[y0, x0] + 0.5:
                surface.set_at((x0, y0), color)
        return

    x_step = dx / steps
    y_step = dy / steps
    z_step = (z_end - z_start) / steps

    x = x0
    y = y0
    z = z_start
    width = z_buffer.shape[1]
    height = z_buffer.shape[0]

    for _ in range(steps + 1):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            if z <= z_buffer[yi, xi] + 0.5:
                surface.set_at((xi, yi), color)
        x += x_step
        y += y_step
        z += z_step

# --- Класс КАМЕРЫ ---
class Camera:
    def __init__(self, distance: float, target: Point3D = Point3D(0, 0, 0)):
        self.distance = distance
        self.target = np.array([target.x, target.y, target.z])
        self.theta = -math.pi / 2
        self.phi = math.pi / 2
        self.up = np.array([0, 1, 0])
    def get_rotation_matrix(self) -> np.ndarray:
        x = self.distance * math.sin(self.phi) * math.cos(self.theta)
        y = self.distance * math.cos(self.phi)
        z = self.distance * math.sin(self.phi) * math.sin(self.theta)
        position = np.array([x, y, z])
        f = self.target - position
        f = f / np.linalg.norm(f)
        r = np.cross(f, self.up)
        if np.linalg.norm(r) < 1e-6: r = np.array([1, 0, 0])
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        M = np.eye(4)
        M[0, :3] = r
        M[1, :3] = -u
        M[2, :3] = f
        return M

# --- Генерация текстуры ---
def generate_checkered_texture(width=256, height=256, tile_size=32):
    surface = pygame.Surface((width, height))
    surface.fill((255, 255, 255))
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            if (x // tile_size + y // tile_size) % 2 == 0:
                pygame.draw.rect(surface, (50, 50, 50), (x, y, tile_size, tile_size))
            else:
                pygame.draw.rect(surface, (200, 50, 50), (x, y, tile_size, tile_size))
    return surface

# --- Основная часть программы ---
def draw_text(surface, text, pos, font, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def main():
    pygame.init()
    screen_width, screen_height = 1000, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("3D Polyhedron Viewer - Textured")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Consolas', 16)
    
    camera_distance = 500
    rotation_speed = 1.0
    move_speed = 10.0
    scale_step = 1.05
    
    camera_obj = Camera(camera_distance)
    use_orbit_camera = False
    camera_rotation = np.eye(4)
    arbitrary_p1 = (0.0, 0.0, 0.0)
    arbitrary_p2 = (100.0, 100.0, 0.0)
    arbitrary_angle_step = 15.0


    try:
    
        checkered_texture = pygame.image.load('texture.jpg') 
        
        checkered_texture = pygame.transform.scale(checkered_texture, (512, 512))
        print("Текстура успешно загружена!")
    except (FileNotFoundError, pygame.error):
        print("Файл текстуры не найден. Генерируем шахматную доску.")
        checkered_texture = generate_checkered_texture()
    texture_enabled = True
    
    polyhedrons = {
        '1': create_tetrahedron,
        '2': create_hexahedron,
        '3': create_octahedron,
        '4': create_icosahedron,
        '5': create_dodecahedron,
        '0': lambda: create_surface_plot(lambda x, y: 50 * np.sin(np.sqrt(x**2 + y**2) / 10), (-50, 50), (-50, 50), 30),
        '9': lambda: create_surface_of_revolution([(0, -100, 0), (40, -80, 0), (60, -20, 0), (40, 60, 0), (0, 100, 0)], 'z', 48)
    }
    
    current_poly_key = '2'
    polyhedron = polyhedrons[current_poly_key]()
    polyhedron.shading_mode = 'flat'
    polyhedron.texture = checkered_texture
    
    auto_rotate = {'x': False, 'y': True, 'z': False}
    projection_mode = 'perspective'
    obj_filename = "model.obj"
    
    current_shading_mode = 'flat'

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    texture_enabled = not texture_enabled
                    polyhedron.texture = checkered_texture if texture_enabled else None

                if '0' <= event.unicode <= '5' or event.unicode == '9':
                    if event.unicode in polyhedrons:
                        current_poly_key = event.unicode
                        polyhedron = polyhedrons[current_poly_key]()
                        polyhedron.shading_mode = current_shading_mode
                        polyhedron.texture = checkered_texture if texture_enabled else None
                
                if event.key == pygame.K_r:
                    polyhedron = polyhedrons[current_poly_key]()
                    polyhedron.shading_mode = current_shading_mode
                    polyhedron.texture = checkered_texture if texture_enabled else None
                
                if event.key == pygame.K_x: auto_rotate['x'] = not auto_rotate['x']
                if event.key == pygame.K_y: auto_rotate['y'] = not auto_rotate['y']
                if event.key == pygame.K_z: auto_rotate['z'] = not auto_rotate['z']
                
                if event.key == pygame.K_6: polyhedron.apply_transform(reflection_matrix('xy'))
                if event.key == pygame.K_7: polyhedron.apply_transform(reflection_matrix('xz'))
                if event.key == pygame.K_8: polyhedron.apply_transform(reflection_matrix('yz'))
                if event.key == pygame.K_p: projection_mode = 'axonometric' if projection_mode == 'perspective' else 'perspective'
                
                if event.key == pygame.K_F1: current_shading_mode = 'flat'; polyhedron.shading_mode = 'flat'
                if event.key == pygame.K_F2: current_shading_mode = 'gouraud'; polyhedron.shading_mode = 'gouraud'
                if event.key == pygame.K_F3: current_shading_mode = 'phong_toon'; polyhedron.shading_mode = 'phong_toon'

                if event.key == pygame.K_k: polyhedron.apply_transform(rotation_about_line(arbitrary_p1, arbitrary_p2, arbitrary_angle_step))
                if event.key == pygame.K_u: polyhedron.apply_transform(rotation_axis_through_center_matrix(polyhedron.get_center(), 'x', 10))
                if event.key == pygame.K_i and not use_orbit_camera: polyhedron.apply_transform(rotation_axis_through_center_matrix(polyhedron.get_center(), 'y', 10))
                if event.key == pygame.K_o: polyhedron.apply_transform(rotation_axis_through_center_matrix(polyhedron.get_center(), 'z', 10))
                if event.key == pygame.K_f: 
                    polyhedron.to_obj(obj_filename)
                    print(f"Модель сохранена в {obj_filename}")
                if event.key == pygame.K_g:
                    try:
                        polyhedron = Polyhedron.from_obj(obj_filename)
                        polyhedron.texture = checkered_texture if texture_enabled else None
                        print(f"Модель загружена из {obj_filename}")
                    except Exception as e: print(f"Ошибка: {e}")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: polyhedron.apply_transform(translation_matrix(0, -move_speed, 0))
        if keys[pygame.K_DOWN]: polyhedron.apply_transform(translation_matrix(0, move_speed, 0))
        if keys[pygame.K_LEFT]: polyhedron.apply_transform(translation_matrix(-move_speed, 0, 0))
        if keys[pygame.K_RIGHT]: polyhedron.apply_transform(translation_matrix(move_speed, 0, 0))
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            c = polyhedron.get_center()
            polyhedron.apply_transform(translation_matrix(c.x, c.y, c.z) @ scale_matrix(scale_step, scale_step, scale_step) @ translation_matrix(-c.x, -c.y, -c.z))
        if keys[pygame.K_MINUS]:
            c = polyhedron.get_center()
            polyhedron.apply_transform(translation_matrix(c.x, c.y, c.z) @ scale_matrix(1/scale_step, 1/scale_step, 1/scale_step) @ translation_matrix(-c.x, -c.y, -c.z))
        if keys[pygame.K_d]: polyhedron.apply_transform(rotation_y_matrix(rotation_speed))
        if keys[pygame.K_a]: polyhedron.apply_transform(rotation_y_matrix(-rotation_speed))
        if keys[pygame.K_w]: polyhedron.apply_transform(rotation_x_matrix(rotation_speed))
        if keys[pygame.K_s]: polyhedron.apply_transform(rotation_x_matrix(-rotation_speed))
        if keys[pygame.K_e]: polyhedron.apply_transform(rotation_z_matrix(rotation_speed))
        if keys[pygame.K_q]: polyhedron.apply_transform(rotation_z_matrix(-rotation_speed))
        if auto_rotate['x']: polyhedron.apply_transform(rotation_x_matrix(rotation_speed / 2))
        if auto_rotate['y']: polyhedron.apply_transform(rotation_y_matrix(rotation_speed / 2))
        if auto_rotate['z']: polyhedron.apply_transform(rotation_z_matrix(rotation_speed / 2))

        camera_changed = False
        if keys[pygame.K_c]: camera_obj.theta -= 0.03; use_orbit_camera = True
        if keys[pygame.K_v]: camera_obj.theta += 0.03; use_orbit_camera = True
        if keys[pygame.K_b]: camera_obj.phi = max(0.1, camera_obj.phi - 0.03); use_orbit_camera = True
        if keys[pygame.K_n]: camera_obj.phi = min(math.pi - 0.1, camera_obj.phi + 0.03); use_orbit_camera = True
        if keys[pygame.K_m]: camera_obj.theta = -math.pi/2; camera_obj.phi = math.pi/2; use_orbit_camera = True
        
        if use_orbit_camera: camera_rotation = camera_obj.get_rotation_matrix()
        else:
            if keys[pygame.K_j]: camera_rotation = rotation_y_matrix(-rotation_speed) @ camera_rotation
            if keys[pygame.K_l]: camera_rotation = rotation_y_matrix(rotation_speed) @ camera_rotation
            if keys[pygame.K_i]: camera_rotation = rotation_x_matrix(-rotation_speed) @ camera_rotation

        screen.fill(polyhedron.bg_color)
        z_buffer = np.full((screen_height, screen_width), np.inf, dtype=float)
        polyhedron.draw(screen, camera_distance, screen_width, screen_height, projection_mode, camera_rotation, z_buffer)
        
        info = [
            polyhedron.get_info(),
            f"Проекция: {projection_mode}",
            f"Шейдинг: {current_shading_mode} (F1/F2/F3)",
            f"Текстура (T): {'Вкл' if texture_enabled else 'Выкл'}",
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
            "T: Вкл/Выкл текстуру",
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

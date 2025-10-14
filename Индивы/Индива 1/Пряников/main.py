import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# --- Типы данных ---
Point = Tuple[float, float]
Triangle = List[Point]

class VertexType(Enum):
    CONVEX = "convex"
    REFLEX = "reflex"  # вогнутая
    EAR = "ear"

@dataclass
class Vertex:
    """Класс для хранения информации о вершине"""
    index: int
    point: Point
    vertex_type: VertexType
    is_ear: bool = False

# --- Вспомогательные геометрические функции ---

def calculate_polygon_area(polygon: List[Point]) -> float:
    """Вычисляет ориентированную площадь полигона (формула шнурков)."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0

def is_ccw(polygon: List[Point]) -> bool:
    """Проверяет, что полигон задан в порядке против часовой стрелки."""
    return calculate_polygon_area(polygon) > 0

def cross_product(p1: Point, p2: Point, p3: Point) -> float:
    """Вычисляет псевдоскалярное произведение векторов."""
    return (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

def is_point_in_triangle(pt: Point, v1: Point, v2: Point, v3: Point) -> bool:
    """Проверяет, находится ли точка pt внутри треугольника."""
    eps = 1e-10  # Небольшой допуск для численной стабильности
    
    d1 = cross_product(v1, v2, pt)
    d2 = cross_product(v2, v3, pt)
    d3 = cross_product(v3, v1, pt)
    
    # Проверяем, что все знаки одинаковы
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def is_simple_polygon(polygon: List[Point]) -> bool:
    """Проверяет, что полигон простой (без самопересечений)."""
    n = len(polygon)
    for i in range(n):
        for j in range(i + 2, n):
            if j == (i + n - 1) % n:  # Смежные рёбра
                continue
            if segments_intersect(polygon[i], polygon[(i+1)%n], 
                                 polygon[j], polygon[(j+1)%n]):
                return False
    return True

def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Проверяет пересечение двух отрезков."""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

# --- Оптимизированный алгоритм Ear Clipping ---

class EarClippingTriangulator:
    def __init__(self, polygon: List[Point], validate: bool = True):
        """
        Инициализация триангулятора.
        :param polygon: список вершин полигона
        :param validate: проверять ли корректность полигона
        """
        if len(polygon) < 3:
            raise ValueError("Полигон должен иметь как минимум 3 вершины.")
        
        self.original_polygon = polygon.copy()
        self.polygon = polygon.copy()
        
        if validate and not is_simple_polygon(self.polygon):
            raise ValueError("Полигон имеет самопересечения.")
        
        # Обеспечиваем CCW ориентацию
        if not is_ccw(self.polygon):
            self.polygon.reverse()
        
        self.vertices = []
        self.triangles = []
        self.triangulation_steps = []  # Для анимации
        
    def classify_vertices(self):
        """Классифицирует вершины на выпуклые и вогнутые."""
        n = len(self.polygon)
        self.vertices = []
        
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            cp = cross_product(self.polygon[prev_idx], 
                             self.polygon[i], 
                             self.polygon[next_idx])
            
            vertex_type = VertexType.CONVEX if cp >= 0 else VertexType.REFLEX
            
            vertex = Vertex(
                index=i,
                point=self.polygon[i],
                vertex_type=vertex_type,
                is_ear=False
            )
            self.vertices.append(vertex)
    
    def is_ear(self, vertex_idx: int) -> bool:
        """Проверяет, является ли вершина "ухом"."""
        n = len(self.vertices)
        if n < 3:
            return False
        
        prev_idx = (vertex_idx - 1) % n
        next_idx = (vertex_idx + 1) % n
        
        v_prev = self.vertices[prev_idx].point
        v_curr = self.vertices[vertex_idx].point
        v_next = self.vertices[next_idx].point
        
        # Вершина должна быть выпуклой
        if self.vertices[vertex_idx].vertex_type != VertexType.CONVEX:
            return False
        
        # Проверяем, что никакие другие вершины не лежат внутри треугольника
        for i, vertex in enumerate(self.vertices):
            if i in (prev_idx, vertex_idx, next_idx):
                continue
            
            # Проверяем только вогнутые вершины (оптимизация)
            if vertex.vertex_type == VertexType.REFLEX:
                if is_point_in_triangle(vertex.point, v_prev, v_curr, v_next):
                    return False
        
        return True
    
    def find_ears(self):
        """Находит все "уши" в текущем полигоне."""
        for i in range(len(self.vertices)):
            if self.vertices[i].vertex_type == VertexType.CONVEX:
                self.vertices[i].is_ear = self.is_ear(i)
                if self.vertices[i].is_ear:
                    self.vertices[i].vertex_type = VertexType.EAR
    
    def triangulate(self) -> List[Triangle]:
        """Выполняет триангуляцию полигона."""
        if len(self.polygon) == 3:
            return [self.polygon]
        
        self.classify_vertices()
        self.find_ears()
        
        iterations = 0
        max_iterations = len(self.polygon) * 2
        
        while len(self.vertices) > 3 and iterations < max_iterations:
            iterations += 1
            
            # Находим первое "ухо"
            ear_found = False
            for i in range(len(self.vertices)):
                if self.vertices[i].is_ear:
                    prev_idx = (i - 1) % len(self.vertices)
                    next_idx = (i + 1) % len(self.vertices)
                    
                    # Создаём треугольник
                    triangle = [
                        self.vertices[prev_idx].point,
                        self.vertices[i].point,
                        self.vertices[next_idx].point
                    ]
                    self.triangles.append(triangle)
                    
                    # Сохраняем шаг для анимации
                    self.triangulation_steps.append({
                        'polygon': [v.point for v in self.vertices],
                        'triangle': triangle,
                        'removed_vertex': i
                    })
                    
                    # Удаляем вершину
                    self.vertices.pop(i)
                    
                    # Обновляем классификацию соседних вершин
                    if len(self.vertices) > 2:
                        new_prev_idx = (prev_idx if prev_idx < i else prev_idx - 1) % len(self.vertices)
                        new_next_idx = (next_idx if next_idx < i else next_idx - 1) % len(self.vertices)
                        
                        # Переклассифицируем соседние вершины
                        for idx in [new_prev_idx, new_next_idx]:
                            if 0 <= idx < len(self.vertices):
                                self.update_vertex_classification(idx)
                    
                    ear_found = True
                    break
            
            if not ear_found:
                print(f"Предупреждение: не найдено ухо на итерации {iterations}")
                break
        
        # Добавляем последний треугольник
        if len(self.vertices) == 3:
            triangle = [v.point for v in self.vertices]
            self.triangles.append(triangle)
            self.triangulation_steps.append({
                'polygon': triangle,
                'triangle': triangle,
                'removed_vertex': -1
            })
        
        return self.triangles
    
    def update_vertex_classification(self, vertex_idx: int):
        """Обновляет классификацию вершины после удаления соседней."""
        n = len(self.vertices)
        if n < 3:
            return
        
        prev_idx = (vertex_idx - 1) % n
        next_idx = (vertex_idx + 1) % n
        
        cp = cross_product(
            self.vertices[prev_idx].point,
            self.vertices[vertex_idx].point,
            self.vertices[next_idx].point
        )
        
        old_type = self.vertices[vertex_idx].vertex_type
        new_type = VertexType.CONVEX if cp >= 0 else VertexType.REFLEX
        
        self.vertices[vertex_idx].vertex_type = new_type
        
        # Проверяем, является ли вершина ухом
        if new_type == VertexType.CONVEX:
            self.vertices[vertex_idx].is_ear = self.is_ear(vertex_idx)
            if self.vertices[vertex_idx].is_ear:
                self.vertices[vertex_idx].vertex_type = VertexType.EAR
        else:
            self.vertices[vertex_idx].is_ear = False

# --- Улучшенная визуализация ---

def plot_triangulation_enhanced(polygon: List[Point], triangles: List[Triangle], 
                               show_numbers: bool = True, save_path: Optional[str] = None):
    """Улучшенная визуализация с номерами вершин и цветовой градацией."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Левый график - исходный полигон
    ax1.set_title("Исходный полигон")
    poly_patch = PolygonPatch(polygon, facecolor='lightblue', 
                              edgecolor='blue', linewidth=2, alpha=0.5)
    ax1.add_patch(poly_patch)
    
    # Добавляем номера вершин
    if show_numbers:
        for i, (x, y) in enumerate(polygon):
            ax1.plot(x, y, 'ro', markersize=8)
            ax1.text(x, y, str(i), fontsize=12, ha='right', va='bottom')
    
    # Правый график - триангуляция
    ax2.set_title(f"Триангуляция ({len(triangles)} треугольников)")
    
    # Цветовая карта для треугольников
    colors = plt.cm.rainbow(np.linspace(0, 1, len(triangles)))
    
    for i, (tri, color) in enumerate(zip(triangles, colors)):
        tri_patch = PolygonPatch(tri, facecolor=color, alpha=0.3, 
                                 edgecolor='black', linewidth=1)
        ax2.add_patch(tri_patch)
        
        # Центр треугольника
        cx = sum(p[0] for p in tri) / 3
        cy = sum(p[1] for p in tri) / 3
        ax2.text(cx, cy, str(i), fontsize=8, ha='center', va='center')
    
    # Настройка осей
    for ax in [ax1, ax2]:
        all_x = [p[0] for p in polygon]
        all_y = [p[1] for p in polygon]
        margin = 0.5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# --- Примеры использования ---

def test_examples():
    """Тестирует алгоритм на различных примерах."""
    
    examples = {
        "Простой треугольник": [(0, 0), (4, 0), (2, 3)],
        
        "Квадрат": [(0, 0), (2, 0), (2, 2), (0, 2)],
        
        "Звезда": [(0, 0), (4, -1), (5, 3), (2, 2), (1, 5), (-2, 3)],
        
        "L-образная фигура": [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)],
        
        "Сложный полигон": [
            (0, 0), (2, 0), (2, 1), (3, 1), (3, 0), (5, 0),
            (5, 3), (3, 3), (3, 2), (2, 2), (2, 3), (0, 3)
        ]
    }
    
    for name, polygon in examples.items():
        print(f"\n{'='*50}")
        print(f"Тестируем: {name}")
        print(f"Вершин: {len(polygon)}")
        
        try:
            triangulator = EarClippingTriangulator(polygon, validate=True)
            triangles = triangulator.triangulate()
            
            print(f"Получено треугольников: {len(triangles)}")
            print(f"Ожидалось треугольников: {len(polygon) - 2}")
            
            if len(triangles) == len(polygon) - 2:
                print("✓ Триангуляция успешна!")
            else:
                print("✗ Неверное количество треугольников!")
            
            # Визуализация
            plot_triangulation_enhanced(polygon, triangles, show_numbers=True)
            
        except Exception as e:
            print(f"✗ Ошибка: {e}")

if __name__ == '__main__':
    # Запускаем тесты
    test_examples()
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import copy

class Point:
    """Класс для представления точки"""
    def __init__(self, x: float, y: float, index: int = -1):
        self.x = x
        self.y = y
        self.index = index
    
    def __repr__(self):
        return f"P{self.index}({self.x:.1f},{self.y:.1f})"

class Polygon:
    """Класс для представления полигона"""
    def __init__(self, vertices: List[Point]):
        self.vertices = vertices
        for i, v in enumerate(vertices):
            v.index = i
    
    def copy(self):
        """Создание копии полигона"""
        new_vertices = [Point(v.x, v.y, v.index) for v in self.vertices]
        return Polygon(new_vertices)

class IntrudingVertexTriangulation:
    """Триангуляция методом вторгающейся вершины"""
    
    def __init__(self, polygon_points: List[Tuple[float, float]]):
        # Создаем полигон из точек
        vertices = [Point(x, y, i) for i, (x, y) in enumerate(polygon_points)]
        self.original_polygon = Polygon(vertices)
        self.triangles = []
        self.steps = []  # Для визуализации шагов
        
    def cross_product(self, o: Point, a: Point, b: Point) -> float:
        """Векторное произведение OA x OB"""
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    
    def is_convex_vertex(self, polygon: Polygon, index: int) -> bool:
        """Проверка, является ли вершина выпуклой"""
        n = len(polygon.vertices)
        prev = polygon.vertices[(index - 1) % n]
        curr = polygon.vertices[index]
        next = polygon.vertices[(index + 1) % n]
        
        # Если следующая точка лежит слева от вектора (prev->curr), то вершина выпуклая
        return self.cross_product(prev, curr, next) > 0
    
    def point_in_triangle(self, p: Point, a: Point, b: Point, c: Point) -> bool:
        """Проверка, находится ли точка внутри треугольника"""
        # Используем барицентрические координаты
        def sign(p1, p2, p3):
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
        
        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def distance_to_line(self, point: Point, line_start: Point, line_end: Point) -> float:
        """Расстояние от точки до прямой"""
        # Векторное произведение дает удвоенную площадь треугольника
        area2 = abs(self.cross_product(line_start, line_end, point))
        # Длина основания
        base = np.sqrt((line_end.x - line_start.x)**2 + (line_end.y - line_start.y)**2)
        if base < 1e-10:
            return 0
        return area2 / base
    
    def find_intruding_vertex(self, polygon: Polygon, vertex_index: int) -> Optional[int]:
        """Поиск вторгающейся вершины для данной выпуклой вершины"""
        n = len(polygon.vertices)
        prev_idx = (vertex_index - 1) % n
        next_idx = (vertex_index + 1) % n
        
        prev_vertex = polygon.vertices[prev_idx]
        curr_vertex = polygon.vertices[vertex_index]
        next_vertex = polygon.vertices[next_idx]
        
        # Ищем вершины, которые находятся внутри треугольника
        intruding_vertices = []
        
        for i, vertex in enumerate(polygon.vertices):
            # Пропускаем вершины самого треугольника
            if i in [prev_idx, vertex_index, next_idx]:
                continue
            
            # Проверяем, находится ли вершина внутри треугольника
            if self.point_in_triangle(vertex, prev_vertex, curr_vertex, next_vertex):
                dist = self.distance_to_line(vertex, prev_vertex, next_vertex)
                intruding_vertices.append((i, dist))
        
        # Если есть вторгающиеся вершины, выбираем наиболее удаленную от прямой
        if intruding_vertices:
            intruding_vertices.sort(key=lambda x: x[1], reverse=True)
            return intruding_vertices[0][0]
        
        return None
    
    def find_closest_convex_vertex(self, polygon: Polygon, start_index: int) -> int:
        """Поиск ближайшей выпуклой вершины"""
        n = len(polygon.vertices)
        start_vertex = polygon.vertices[start_index]
        
        min_dist = float('inf')
        closest_index = start_index
        
        for i in range(n):
            if i == start_index:
                continue
            
            if self.is_convex_vertex(polygon, i):
                vertex = polygon.vertices[i]
                dist = (vertex.x - start_vertex.x)**2 + (vertex.y - start_vertex.y)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i
        
        return closest_index
    
    def split_polygon(self, polygon: Polygon, v1_idx: int, v2_idx: int) -> Tuple[Polygon, Polygon]:
        """Разбиение полигона на два по диагонали"""
        n = len(polygon.vertices)
        
        # Убеждаемся, что v1_idx < v2_idx
        if v1_idx > v2_idx:
            v1_idx, v2_idx = v2_idx, v1_idx
        
        # Первый полигон: от v1 до v2 по часовой стрелке
        poly1_vertices = []
        i = v1_idx
        while i != v2_idx:
            poly1_vertices.append(polygon.vertices[i])
            i = (i + 1) % n
        poly1_vertices.append(polygon.vertices[v2_idx])
        
        # Второй полигон: от v2 до v1 по часовой стрелке
        poly2_vertices = []
        i = v2_idx
        while i != v1_idx:
            poly2_vertices.append(polygon.vertices[i])
            i = (i + 1) % n
        poly2_vertices.append(polygon.vertices[v1_idx])
        
        return Polygon(poly1_vertices), Polygon(poly2_vertices)
    
    def triangulate_recursive(self, polygon: Polygon):
        """Рекурсивная триангуляция полигона"""
        n = len(polygon.vertices)
        
        # Базовый случай - полигон уже треугольник
        if n == 3:
            self.triangles.append(polygon.vertices[:])
            self.steps.append({
                'type': 'triangle',
                'polygon': polygon.copy(),
                'triangle': polygon.vertices[:]
            })
            return
        
        # Ищем выпуклую вершину (начинаем с первой)
        start_vertex = 0
        convex_vertex = self.find_closest_convex_vertex(polygon, start_vertex)
        
        # Ищем вторгающуюся вершину
        intruding = self.find_intruding_vertex(polygon, convex_vertex)
        
        if intruding is None:
            # Нет вторгающихся вершин - отсекаем треугольник
            n = len(polygon.vertices)
            prev_idx = (convex_vertex - 1) % n
            next_idx = (convex_vertex + 1) % n
            
            triangle = [
                polygon.vertices[prev_idx],
                polygon.vertices[convex_vertex],
                polygon.vertices[next_idx]
            ]
            self.triangles.append(triangle)
            
            self.steps.append({
                'type': 'cut_triangle',
                'polygon': polygon.copy(),
                'vertex': convex_vertex,
                'triangle': triangle
            })
            
            # Создаем новый полигон без выпуклой вершины
            new_vertices = []
            for i in range(n):
                if i != convex_vertex:
                    new_vertices.append(polygon.vertices[i])
            
            new_polygon = Polygon(new_vertices)
            self.triangulate_recursive(new_polygon)
            
        else:
            # Есть вторгающаяся вершина - разбиваем полигон
            self.steps.append({
                'type': 'split',
                'polygon': polygon.copy(),
                'vertex': convex_vertex,
                'intruding': intruding,
                'edge': [polygon.vertices[convex_vertex], polygon.vertices[intruding]]
            })
            
            # Разбиваем полигон на два
            poly1, poly2 = self.split_polygon(polygon, convex_vertex, intruding)
            
            # Рекурсивно триангулируем оба полигона
            self.triangulate_recursive(poly1)
            self.triangulate_recursive(poly2)
    
    def triangulate(self):
        """Основной метод триангуляции"""
        self.triangles = []
        self.steps = []
        
        # Начинаем с копии исходного полигона
        polygon = self.original_polygon.copy()
        
        self.steps.append({
            'type': 'initial',
            'polygon': polygon.copy()
        })
        
        self.triangulate_recursive(polygon)
        
        return self.triangles
    
    def visualize(self):
        """Визуализация результата триангуляции"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Левый график - исходный полигон
        ax1 = axes[0]
        ax1.set_title('Исходный полигон', fontsize=14, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Рисуем полигон
        vertices = self.original_polygon.vertices
        poly_x = [v.x for v in vertices] + [vertices[0].x]
        poly_y = [v.y for v in vertices] + [vertices[0].y]
        ax1.plot(poly_x, poly_y, 'b-', linewidth=2)
        
        # Отмечаем вершины
        for v in vertices:
            ax1.plot(v.x, v.y, 'ro', markersize=8)
            ax1.annotate(f'{v.index}', (v.x, v.y), 
                        xytext=(5, 5), textcoords='offset points')
            
            # Отмечаем выпуклые вершины
            if self.is_convex_vertex(self.original_polygon, v.index):
                ax1.plot(v.x, v.y, 'go', markersize=10, alpha=0.5)
        
        # Правый график - триангуляция
        ax2 = axes[1]
        ax2.set_title('Результат триангуляции', fontsize=14, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Рисуем треугольники
        for triangle in self.triangles:
            tri_x = [v.x for v in triangle] + [triangle[0].x]
            tri_y = [v.y for v in triangle] + [triangle[0].y]
            ax2.plot(tri_x, tri_y, 'b-', linewidth=1)
            ax2.fill(tri_x, tri_y, alpha=0.2)
        
        # Отмечаем вершины
        for v in self.original_polygon.vertices:
            ax2.plot(v.x, v.y, 'ro', markersize=8)
            ax2.annotate(f'{v.index}', (v.x, v.y), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_steps(self):
        """Пошаговая визуализация процесса триангуляции"""
        n_steps = min(len(self.steps), 12)  # Ограничиваем количество шагов
        
        if n_steps <= 4:
            rows, cols = 2, 2
        elif n_steps <= 6:
            rows, cols = 2, 3
        elif n_steps <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 3, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten() if n_steps > 1 else [axes]
        
        for i in range(min(n_steps, len(axes))):
            ax = axes[i]
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            if i >= len(self.steps):
                ax.axis('off')
                continue
            
            step = self.steps[i]
            
            # Заголовок в зависимости от типа шага
            if step['type'] == 'initial':
                ax.set_title(f'Шаг {i}: Исходный полигон', fontsize=10)
            elif step['type'] == 'cut_triangle':
                ax.set_title(f'Шаг {i}: Отсечение треугольника', fontsize=10)
            elif step['type'] == 'split':
                ax.set_title(f'Шаг {i}: Разбиение полигона', fontsize=10)
            elif step['type'] == 'triangle':
                ax.set_title(f'Шаг {i}: Добавление треугольника', fontsize=10)
            
            # Рисуем текущий полигон
            if 'polygon' in step:
                polygon = step['polygon']
                vertices = polygon.vertices
                poly_x = [v.x for v in vertices] + [vertices[0].x]
                poly_y = [v.y for v in vertices] + [vertices[0].y]
                ax.plot(poly_x, poly_y, 'b-', linewidth=1, alpha=0.5)
            
            # Выделяем элементы в зависимости от типа шага
            if step['type'] == 'cut_triangle' and 'triangle' in step:
                triangle = step['triangle']
                tri_x = [v.x for v in triangle] + [triangle[0].x]
                tri_y = [v.y for v in triangle] + [triangle[0].y]
                ax.fill(tri_x, tri_y, 'green', alpha=0.3)
                ax.plot(tri_x, tri_y, 'g-', linewidth=2)
                
            elif step['type'] == 'split' and 'edge' in step:
                edge = step['edge']
                ax.plot([edge[0].x, edge[1].x], [edge[0].y, edge[1].y], 
                       'r-', linewidth=3)
                
            elif step['type'] == 'triangle' and 'triangle' in step:
                triangle = step['triangle']
                tri_x = [v.x for v in triangle] + [triangle[0].x]
                tri_y = [v.y for v in triangle] + [triangle[0].y]
                ax.fill(tri_x, tri_y, 'blue', alpha=0.3)
                ax.plot(tri_x, tri_y, 'b-', linewidth=2)
        
        # Скрываем неиспользуемые графики
        for i in range(n_steps, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def demonstrate():
    """Демонстрация работы алгоритма"""
    
    print("=" * 60)
    print("ТРИАНГУЛЯЦИЯ МЕТОДОМ ВТОРГАЮЩЕЙСЯ ВЕРШИНЫ")
    print("=" * 60)
    
    # Пример полигона
    polygon_points = [
        (1, 1),
        (4, 0.5),
        (6, 2),
        (5, 4),
        (3, 3.5),
        (2, 5),
        (0, 3)
    ]
    
    print("\nВершины полигона:")
    for i, (x, y) in enumerate(polygon_points):
        print(f"  Вершина {i}: ({x}, {y})")
    
    # Создаем триангулятор
    triangulator = IntrudingVertexTriangulation(polygon_points)
    
    # Выполняем триангуляцию
    triangles = triangulator.triangulate()
    
    print(f"\nРезультат триангуляции:")
    print(f"  Количество треугольников: {len(triangles)}")
    print(f"  Ожидаемое количество: {len(polygon_points) - 2}")
    
    for i, triangle in enumerate(triangles):
        indices = [v.index for v in triangle]
        print(f"  Треугольник {i}: вершины {indices}")
    
    # Визуализация
    print("\nВизуализация результата...")
    triangulator.visualize()
    
    print("\nПошаговая визуализация...")
    triangulator.visualize_steps()

if __name__ == "__main__":
    demonstrate()
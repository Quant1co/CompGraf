import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import List, Tuple, Optional
import copy

class Point:
    """Класс для представления точки в 2D пространстве"""
    def __init__(self, x: float, y: float, index: int = -1):
        self.x = x          # координата X точки
        self.y = y          # координата Y точки
        self.index = index  # индекс точки в исходном полигоне (для идентификации)
    
    def __repr__(self):
        # Строковое представление для удобной отладки: P0(1.5,2.3)
        return f"P{self.index}({self.x:.1f},{self.y:.1f})"

class Polygon:
    """Класс для представления полигона как упорядоченного списка вершин"""
    def __init__(self, vertices: List[Point]):
        self.vertices = vertices  # список вершин в порядке обхода полигона
        # Присваиваем каждой вершине её порядковый номер в полигоне
        for i, v in enumerate(vertices):
            v.index = i
    
    def copy(self):
        """Создание глубокой копии полигона для сохранения промежуточных состояний"""
        # Создаём новые объекты Point, чтобы изменения не затрагивали оригинал
        new_vertices = [Point(v.x, v.y, v.index) for v in self.vertices]
        return Polygon(new_vertices)

class IntrudingVertexTriangulation:
    """
    Класс для триангуляции простого полигона методом вторгающейся вершины.
    
    Алгоритм:
    1. Находим выпуклую вершину
    2. Проверяем, есть ли вторгающиеся вершины в треугольнике (i-1, i, i+1)
    3. Если нет - отсекаем треугольник
    4. Если есть - разбиваем полигон диагональю
    """
    
    def __init__(self, polygon_points: List[Tuple[float, float]]):
        # Преобразуем список координат в объекты Point с индексами
        vertices = [Point(x, y, i) for i, (x, y) in enumerate(polygon_points)]
        self.original_polygon = Polygon(vertices)  # сохраняем исходный полигон
        self.triangles = []  # список результирующих треугольников
        self.steps = []      # история шагов для визуализации процесса
        
    def cross_product(self, o: Point, a: Point, b: Point) -> float:
        """
        Векторное произведение векторов OA и OB.
        
        Математика: (OA × OB)_z = (a.x - o.x)*(b.y - o.y) - (a.y - o.y)*(b.x - o.x)
        
        Результат:
        > 0: точка B слева от вектора OA (поворот против часовой стрелки)
        < 0: точка B справа от вектора OA (поворот по часовой стрелке)
        = 0: точки O, A, B коллинеарны
        """
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    
    def is_convex_vertex(self, polygon: Polygon, index: int) -> bool:
        """
        Проверка, является ли вершина выпуклой.
        
        Вершина выпуклая, если при обходе полигона против часовой стрелки
        поворот в этой вершине происходит влево (против часовой стрелки).
        
        Берём три последовательные вершины: prev -> curr -> next
        Если next слева от вектора (prev->curr), то curr - выпуклая
        """
        n = len(polygon.vertices)
        # Получаем три последовательные вершины с учётом цикличности
        prev = polygon.vertices[(index - 1) % n]  # предыдущая вершина
        curr = polygon.vertices[index]            # текущая проверяемая вершина
        next = polygon.vertices[(index + 1) % n]  # следующая вершина
        
        # Векторное произведение > 0 означает поворот влево (выпуклая вершина)
        return self.cross_product(prev, curr, next) > 0
    
    def point_in_triangle(self, p: Point, a: Point, b: Point, c: Point) -> bool:
        """
        Проверка, находится ли точка p внутри треугольника ABC.
        
        Используем метод ориентации:
        Точка внутри треугольника, если она находится с одной стороны
        относительно всех трёх рёбер треугольника.
        
        sign() вычисляет с какой стороны от прямой находится точка
        """
        def sign(p1, p2, p3):
            # Вычисляем ориентацию точки p1 относительно прямой (p2,p3)
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
        
        # Проверяем положение точки относительно каждой стороны треугольника
        d1 = sign(p, a, b)  # относительно стороны AB
        d2 = sign(p, b, c)  # относительно стороны BC
        d3 = sign(p, c, a)  # относительно стороны CA
        
        # Проверяем, есть ли отрицательные и положительные значения
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        # Точка внутри, если все знаки одинаковые (все + или все -)
        return not (has_neg and has_pos)
    
    def distance_to_line(self, point: Point, line_start: Point, line_end: Point) -> float:
        """
        Расстояние от точки до прямой, заданной двумя точками.
        
        Формула через площадь:
        - Векторное произведение даёт удвоенную площадь треугольника
        - Площадь = 0.5 * основание * высота
        - Следовательно: высота = удвоенная_площадь / основание
        """
        # Удвоенная площадь треугольника через векторное произведение
        area2 = abs(self.cross_product(line_start, line_end, point))
        
        # Длина основания (расстояние между концами отрезка)
        base = np.sqrt((line_end.x - line_start.x)**2 + (line_end.y - line_start.y)**2)
        
        # Избегаем деления на ноль для вырожденного случая
        if base < 1e-10:
            return 0
            
        # Расстояние = удвоенная площадь / основание
        return area2 / base
    
    def find_intruding_vertex(self, polygon: Polygon, vertex_index: int) -> Optional[int]:
        """
        Поиск вторгающейся вершины для выпуклой вершины vertex_index.
        
        Вторгающаяся вершина - это вершина полигона, которая находится
        внутри треугольника, образованного вершиной и её соседями.
        
        Из всех вторгающихся выбираем самую удалённую от диагонали (i-1,i+1),
        чтобы избежать самопересечений при разбиении.
        """
        n = len(polygon.vertices)
        # Индексы вершин треугольника (с учётом цикличности)
        prev_idx = (vertex_index - 1) % n
        next_idx = (vertex_index + 1) % n
        
        # Получаем вершины треугольника
        prev_vertex = polygon.vertices[prev_idx]
        curr_vertex = polygon.vertices[vertex_index]
        next_vertex = polygon.vertices[next_idx]
        
        # Список для хранения вторгающихся вершин и их расстояний до диагонали
        intruding_vertices = []
        
        # Проверяем все вершины полигона
        for i, vertex in enumerate(polygon.vertices):
            # Пропускаем вершины самого треугольника
            if i in [prev_idx, vertex_index, next_idx]:
                continue
            
            # Проверяем, находится ли вершина внутри треугольника
            if self.point_in_triangle(vertex, prev_vertex, curr_vertex, next_vertex):
                # Вычисляем расстояние от вершины до диагонали (prev_vertex, next_vertex)
                dist = self.distance_to_line(vertex, prev_vertex, next_vertex)
                intruding_vertices.append((i, dist))  # сохраняем индекс и расстояние
        
        # Если есть вторгающиеся вершины
        if intruding_vertices:
            # Сортируем по убыванию расстояния и выбираем самую удалённую
            # Это гарантирует корректное разбиение без самопересечений
            intruding_vertices.sort(key=lambda x: x[1], reverse=True)
            return intruding_vertices[0][0]  # возвращаем индекс самой удалённой
        
        return None  # вторгающихся вершин нет
    
    def find_closest_convex_vertex(self, polygon: Polygon, start_index: int) -> int:
        """
        Поиск ближайшей выпуклой вершины к заданной стартовой вершине.
        
        Это нужно для выбора вершины, с которой начнём обработку.
        Выбор ближайшей помогает получить более равномерную триангуляцию.
        """
        n = len(polygon.vertices)
        start_vertex = polygon.vertices[start_index]
        
        min_dist = float('inf')  # начальное значение для поиска минимума
        closest_index = start_index  # по умолчанию возвращаем стартовую
        
        # Проверяем все вершины полигона
        for i in range(n):
            if i == start_index:
                continue  # пропускаем стартовую вершину
            
            # Проверяем, является ли вершина выпуклой
            if self.is_convex_vertex(polygon, i):
                vertex = polygon.vertices[i]
                # Вычисляем квадрат расстояния (не извлекаем корень для оптимизации)
                dist = (vertex.x - start_vertex.x)**2 + (vertex.y - start_vertex.y)**2
                
                # Обновляем минимум, если нашли более близкую выпуклую вершину
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i
        
        return closest_index
    
    def split_polygon(self, polygon: Polygon, v1_idx: int, v2_idx: int) -> Tuple[Polygon, Polygon]:
        """
        Разбиение полигона на два меньших полигона диагональю между вершинами v1 и v2.
        
        Процесс:
        1. Диагональ соединяет вершины v1_idx и v2_idx
        2. Первый полигон: путь от v1 до v2 по контуру + диагональ
        3. Второй полигон: путь от v2 до v1 по контуру + диагональ
        
        Важно: обе вершины диагонали включаются в оба полигона (дублируются)
        """
        n = len(polygon.vertices)
        
        # Упорядочиваем индексы для удобства (меньший первым)
        if v1_idx > v2_idx:
            v1_idx, v2_idx = v2_idx, v1_idx
        
        # === Первый полигон: от v1 до v2 по часовой стрелке ===
        poly1_vertices = []
        i = v1_idx
        # Идём по контуру от v1 до v2
        while i != v2_idx:
            poly1_vertices.append(polygon.vertices[i])
            i = (i + 1) % n  # следующая вершина с учётом цикличности
        # Добавляем конечную вершину v2
        poly1_vertices.append(polygon.vertices[v2_idx])
        
        # === Второй полигон: от v2 до v1 по часовой стрелке ===
        poly2_vertices = []
        i = v2_idx
        # Идём по контуру от v2 до v1
        while i != v1_idx:
            poly2_vertices.append(polygon.vertices[i])
            i = (i + 1) % n  # следующая вершина с учётом цикличности
        # Добавляем конечную вершину v1
        poly2_vertices.append(polygon.vertices[v1_idx])
        
        # Возвращаем два новых полигона
        return Polygon(poly1_vertices), Polygon(poly2_vertices)
    
    def triangulate_recursive(self, polygon: Polygon):
        """
        Рекурсивная триангуляция полигона.
        
        Алгоритм:
        1. Базовый случай: если полигон - треугольник, добавляем его в результат
        2. Находим выпуклую вершину
        3. Проверяем наличие вторгающихся вершин
        4. Если нет вторгающихся:
           - Отсекаем треугольник (i-1, i, i+1)
           - Удаляем вершину i из полигона
           - Рекурсивно обрабатываем оставшийся полигон
        5. Если есть вторгающиеся:
           - Проводим диагональ к самой удалённой вторгающейся вершине
           - Разбиваем полигон на два
           - Рекурсивно обрабатываем оба полигона
        """
        n = len(polygon.vertices)
        
        # === БАЗОВЫЙ СЛУЧАЙ: полигон уже является треугольником ===
        if n == 3:
            # Добавляем треугольник в результат
            self.triangles.append(polygon.vertices[:])
            # Сохраняем шаг для визуализации
            self.steps.append({
                'type': 'triangle',
                'polygon': polygon.copy(),
                'triangle': polygon.vertices[:]
            })
            return  # завершаем рекурсию
        
        # === РЕКУРСИВНЫЙ СЛУЧАЙ: полигон имеет больше 3 вершин ===
        
        # Шаг 1: Находим выпуклую вершину (начинаем поиск с вершины 0)
        start_vertex = 0
        convex_vertex = self.find_closest_convex_vertex(polygon, start_vertex)
        
        # Шаг 2: Ищем вторгающиеся вершины для найденной выпуклой вершины
        intruding = self.find_intruding_vertex(polygon, convex_vertex)
        
        if intruding is None:
            # === СЛУЧАЙ 1: Нет вторгающихся вершин - можно отсечь треугольник ===
            
            n = len(polygon.vertices)
            # Индексы вершин треугольника для отсечения
            prev_idx = (convex_vertex - 1) % n  # предыдущая вершина
            next_idx = (convex_vertex + 1) % n  # следующая вершина
            
            # Формируем треугольник из трёх последовательных вершин
            triangle = [
                polygon.vertices[prev_idx],
                polygon.vertices[convex_vertex],
                polygon.vertices[next_idx]
            ]
            # Добавляем треугольник в результат
            self.triangles.append(triangle)
            
            # Сохраняем шаг для визуализации
            self.steps.append({
                'type': 'cut_triangle',
                'polygon': polygon.copy(),
                'vertex': convex_vertex,
                'triangle': triangle
            })
            
            # Создаём новый полигон без выпуклой вершины
            # (удаляем вершину convex_vertex из полигона)
            new_vertices = []
            for i in range(n):
                if i != convex_vertex:  # копируем все вершины кроме удаляемой
                    new_vertices.append(polygon.vertices[i])
            
            # Создаём новый полигон с уменьшенным числом вершин
            new_polygon = Polygon(new_vertices)
            # Рекурсивно триангулируем оставшийся полигон
            self.triangulate_recursive(new_polygon)
            
        else:
            # === СЛУЧАЙ 2: Есть вторгающаяся вершина - разбиваем полигон ===
            
            # Сохраняем шаг разбиения для визуализации
            self.steps.append({
                'type': 'split',
                'polygon': polygon.copy(),
                'vertex': convex_vertex,
                'intruding': intruding,
                'edge': [polygon.vertices[convex_vertex], polygon.vertices[intruding]]
            })
            
            # Разбиваем полигон диагональю на два меньших полигона
            poly1, poly2 = self.split_polygon(polygon, convex_vertex, intruding)
            
            # Рекурсивно триангулируем оба получившихся полигона
            self.triangulate_recursive(poly1)
            self.triangulate_recursive(poly2)
    
    def triangulate(self):
        """
        Основной публичный метод для запуска триангуляции.
        Инициализирует процесс и возвращает список треугольников.
        """
        # Очищаем результаты предыдущих запусков
        self.triangles = []
        self.steps = []
        
        # Создаём копию исходного полигона для обработки
        # (чтобы не изменять оригинал)
        polygon = self.original_polygon.copy()
        
        # Сохраняем начальное состояние для визуализации
        self.steps.append({
            'type': 'initial',
            'polygon': polygon.copy()
        })
        
        # Запускаем рекурсивную триангуляцию
        self.triangulate_recursive(polygon)
        
        # Возвращаем список полученных треугольников
        return self.triangles
    
    def visualize(self):
        """
        Визуализация результата триангуляции.
        Показывает исходный полигон и результирующую триангуляцию.
        """
        # Создаём фигуру с двумя графиками рядом
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # === ЛЕВЫЙ ГРАФИК: Исходный полигон ===
        ax1 = axes[0]
        ax1.set_title('Исходный полигон', fontsize=14, fontweight='bold')
        ax1.set_aspect('equal')  # одинаковый масштаб по осям
        ax1.grid(True, alpha=0.3)
        
        # Рисуем контур полигона
        vertices = self.original_polygon.vertices
        poly_x = [v.x for v in vertices] + [vertices[0].x]  # замыкаем контур
        poly_y = [v.y for v in vertices] + [vertices[0].y]
        ax1.plot(poly_x, poly_y, 'b-', linewidth=2)
        
        # Отмечаем все вершины
        for v in vertices:
            ax1.plot(v.x, v.y, 'ro', markersize=8)  # красная точка
            # Подписываем индекс вершины
            ax1.annotate(f'{v.index}', (v.x, v.y), 
                        xytext=(5, 5), textcoords='offset points')
            
            # Дополнительно выделяем выпуклые вершины зелёным
            if self.is_convex_vertex(self.original_polygon, v.index):
                ax1.plot(v.x, v.y, 'go', markersize=10, alpha=0.5)
        
        # === ПРАВЫЙ ГРАФИК: Результат триангуляции ===
        ax2 = axes[1]
        ax2.set_title('Результат триангуляции', fontsize=14, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Рисуем все треугольники
        for triangle in self.triangles:
            # Координаты вершин треугольника
            tri_x = [v.x for v in triangle] + [triangle[0].x]  # замыкаем треугольник
            tri_y = [v.y for v in triangle] + [triangle[0].y]
            ax2.plot(tri_x, tri_y, 'b-', linewidth=1)  # контур треугольника
            ax2.fill(tri_x, tri_y, alpha=0.2)  # полупрозрачная заливка
        
        # Отмечаем вершины
        for v in self.original_polygon.vertices:
            ax2.plot(v.x, v.y, 'ro', markersize=8)
            ax2.annotate(f'{v.index}', (v.x, v.y), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_steps(self):
        """
        Пошаговая визуализация процесса триангуляции.
        Показывает промежуточные шаги алгоритма.
        """
        # Ограничиваем количество шагов для удобства отображения
        n_steps = min(len(self.steps), 12)
        
        # Определяем размер сетки для подграфиков
        if n_steps <= 4:
            rows, cols = 2, 2
        elif n_steps <= 6:
            rows, cols = 2, 3
        elif n_steps <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 3, 4
        
        # Создаём сетку подграфиков
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten() if n_steps > 1 else [axes]  # преобразуем в одномерный массив
        
        # Отображаем каждый шаг
        for i in range(min(n_steps, len(axes))):
            ax = axes[i]
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Проверяем, есть ли данные для этого шага
            if i >= len(self.steps):
                ax.axis('off')
                continue
            
            step = self.steps[i]
            
            # Устанавливаем заголовок в зависимости от типа шага
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
                # Выделяем отсекаемый треугольник зелёным
                triangle = step['triangle']
                tri_x = [v.x for v in triangle] + [triangle[0].x]
                tri_y = [v.y for v in triangle] + [triangle[0].y]
                ax.fill(tri_x, tri_y, 'green', alpha=0.3)
                ax.plot(tri_x, tri_y, 'g-', linewidth=2)
                
            elif step['type'] == 'split' and 'edge' in step:
                # Выделяем диагональ разбиения красным
                edge = step['edge']
                ax.plot([edge[0].x, edge[1].x], [edge[0].y, edge[1].y], 
                       'r-', linewidth=3)
                
            elif step['type'] == 'triangle' and 'triangle' in step:
                # Выделяем финальный треугольник синим
                triangle = step['triangle']
                tri_x = [v.x for v in triangle] + [triangle[0].x]
                tri_y = [v.y for v in triangle] + [triangle[0].y]
                ax.fill(tri_x, tri_y, 'blue', alpha=0.3)
                ax.plot(tri_x, tri_y, 'b-', linewidth=2)
        
        # Скрываем неиспользуемые подграфики
        for i in range(n_steps, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


class InteractivePolygonBuilder:
    """
    Интерактивный построитель полигона.
    Позволяет создавать полигон кликами мыши и триангулировать его.
    """
    
    def __init__(self):
        self.points = []  # список точек полигона
        self.fig = None
        self.ax = None
        self.line = None  # линия полигона
        self.point_markers = None  # маркеры точек
        self.is_closed = False  # флаг закрытия полигона
        self.triangulator = None  # объект триангулятора
        self.triangulation_lines = []  # список линий триангуляции
        self.triangulation_patches = []  # список патчей (заливок) триангуляции
        self.annotations = []  # список аннотаций для точек
        
    def start(self):
        """Запуск интерактивного режима построения полигона"""
        
        # Создаём окно
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title('ИНТЕРАКТИВНОЕ ПОСТРОЕНИЕ ПОЛИГОНА\n'
                          'Левый клик - добавить точку | Правый клик - замкнуть полигон', 
                          fontsize=12, fontweight='bold')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Устанавливаем границы области
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 11)
        
        # Инициализируем пустые линии
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)
        self.point_markers, = self.ax.plot([], [], 'ro', markersize=8)
        
        # Создаём кнопки управления
        self._create_buttons()
        
        # Подключаем обработчики событий
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Инструкция
        self.info_text = self.ax.text(0.02, 0.98, 
                                      'Точек: 0\nСтатус: Построение', 
                                      transform=self.ax.transAxes,
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.show()
    
    def _create_buttons(self):
        """Создание кнопок управления"""
        
        # Кнопка "Очистить"
        ax_clear = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.btn_clear = Button(ax_clear, 'Очистить')
        self.btn_clear.on_clicked(self.clear)
        
        # Кнопка "Триангулировать"
        ax_triangulate = plt.axes([0.81, 0.05, 0.15, 0.04])
        self.btn_triangulate = Button(ax_triangulate, 'Триангулировать')
        self.btn_triangulate.on_clicked(self.triangulate)
        
        # Кнопка "Демо"
        ax_demo = plt.axes([0.59, 0.05, 0.1, 0.04])
        self.btn_demo = Button(ax_demo, 'Демо')
        self.btn_demo.on_clicked(self.load_demo)
    
    def on_click(self, event):
        """Обработчик клика мыши"""
        
        # Проверяем, что клик был в области графика
        if event.inaxes != self.ax:
            return
        
        # Если полигон уже замкнут, не добавляем новые точки
        if self.is_closed:
            return
        
        # Левый клик - добавляем точку
        if event.button == 1:  # левая кнопка мыши
            self.add_point(event.xdata, event.ydata)
        
        # Правый клик - замыкаем полигон
        elif event.button == 3:  # правая кнопка мыши
            self.close_polygon()
    
    def add_point(self, x, y):
        """Добавление новой точки к полигону"""
        
        # Добавляем точку
        self.points.append((x, y))
        
        # Обновляем визуализацию
        self.update_display()
        
        # Обновляем информацию
        self.info_text.set_text(f'Точек: {len(self.points)}\nСтатус: Построение')
    
    def close_polygon(self):
        """Замыкание полигона"""
        
        if len(self.points) < 3:
            self.info_text.set_text(f'Точек: {len(self.points)}\n'
                                   f'Статус: Нужно минимум 3 точки!')
            return
        
        # Проверяем на самопересечения
        if self.has_self_intersections():
            self.info_text.set_text(f'Точек: {len(self.points)}\n'
                                   f'Статус: Полигон имеет самопересечения!')
            return
        
        self.is_closed = True
        
        # Добавляем первую точку в конец для замыкания визуально
        if self.points and self.points[0] != self.points[-1]:
            xs = [p[0] for p in self.points] + [self.points[0][0]]
            ys = [p[1] for p in self.points] + [self.points[0][1]]
            self.line.set_data(xs, ys)
        
        # Обновляем информацию
        self.info_text.set_text(f'Точек: {len(self.points)}\n'
                               f'Статус: Полигон замкнут')
        
        self.fig.canvas.draw()
    
    def has_self_intersections(self):
        """Проверка на самопересечения полигона"""
        
        n = len(self.points)
        if n < 4:
            return False
        
        # Проверяем пересечение каждого ребра с каждым
        for i in range(n):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % n]
            
            for j in range(i + 2, n):
                # Не проверяем соседние рёбра
                if (j == (i - 1) % n) or ((j + 1) % n == i):
                    continue
                
                p3 = self.points[j]
                p4 = self.points[(j + 1) % n]
                
                if self.segments_intersect(p1, p2, p3, p4):
                    return True
        
        return False
    
    def segments_intersect(self, p1, p2, p3, p4):
        """Проверка пересечения двух отрезков"""
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def update_display(self):
        """Обновление отображения полигона"""
        
        if not self.points:
            return
        
        # Обновляем линию
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        
        if self.is_closed and len(self.points) > 0:
            xs.append(self.points[0][0])
            ys.append(self.points[0][1])
        
        self.line.set_data(xs, ys)
        
        # Обновляем точки
        self.point_markers.set_data([p[0] for p in self.points], 
                                   [p[1] for p in self.points])
        
        # Удаляем старые аннотации
        for ann in self.annotations:
            ann.remove()
        self.annotations = []
        
        # Добавляем новые аннотации к точкам
        for i, (x, y) in enumerate(self.points):
            ann = self.ax.annotate(f'{i}', (x, y), 
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=10)
            self.annotations.append(ann)
        
        self.fig.canvas.draw()
    
    def clear(self, event=None):
        """Очистка всех точек и начало заново"""
        
        # Очищаем данные
        self.points = []
        self.is_closed = False
        self.line.set_data([], [])
        self.point_markers.set_data([], [])
        
        # Удаляем все линии триангуляции
        for line in self.triangulation_lines:
            line.remove()
        self.triangulation_lines = []
        
        # Удаляем все заливки триангуляции
        for patch in self.triangulation_patches:
            patch.remove()
        self.triangulation_patches = []
        
        # Удаляем все аннотации
        for ann in self.annotations:
            ann.remove()
        self.annotations = []
        
        # Очищаем все патчи (на всякий случай)
        for patch in self.ax.patches[:]:
            patch.remove()
        
        # Удаляем все дополнительные линии (кроме основных)
        lines_to_remove = []
        for line in self.ax.lines:
            if line not in [self.line, self.point_markers]:
                lines_to_remove.append(line)
        for line in lines_to_remove:
            line.remove()
        
        self.info_text.set_text('Точек: 0\nСтатус: Построение')
        
        self.fig.canvas.draw()
    
    def load_demo(self, event=None):
        """Загрузка демонстрационного полигона"""
        
        self.clear()
        
        # Демонстрационный полигон
        demo_points = [
            (2, 2),
            (5, 1),
            (8, 2),
            (9, 5),
            (7, 7),
            (4, 8),
            (1, 6),
            (1, 4)
        ]
        
        for x, y in demo_points:
            self.add_point(x, y)
        
        self.close_polygon()
    
    def triangulate(self, event=None):
        """Триангуляция построенного полигона"""
        
        if not self.is_closed:
            self.info_text.set_text(f'Точек: {len(self.points)}\n'
                                   f'Статус: Сначала замкните полигон!')
            return
        
        if len(self.points) < 3:
            self.info_text.set_text(f'Точек: {len(self.points)}\n'
                                   f'Статус: Недостаточно точек!')
            return
        
        try:
            # Очищаем предыдущую триангуляцию если была
            for line in self.triangulation_lines:
                line.remove()
            self.triangulation_lines = []
            
            for patch in self.triangulation_patches:
                patch.remove()
            self.triangulation_patches = []
            
            # Создаём триангулятор
            self.triangulator = IntrudingVertexTriangulation(self.points)
            
            # Выполняем триангуляцию
            triangles = self.triangulator.triangulate()
            
            # Отображаем треугольники
            for triangle in triangles:
                tri_x = [v.x for v in triangle] + [triangle[0].x]
                tri_y = [v.y for v in triangle] + [triangle[0].y]
                
                # Рисуем контур треугольника и сохраняем ссылку
                line, = self.ax.plot(tri_x, tri_y, 'g-', linewidth=1, alpha=0.7)
                self.triangulation_lines.append(line)
                
                # Заливаем треугольник случайным цветом и сохраняем ссылку
                color = np.random.rand(3,)
                patch = plt.Polygon([(v.x, v.y) for v in triangle], 
                                   alpha=0.2, color=color)
                self.ax.add_patch(patch)
                self.triangulation_patches.append(patch)
            
            self.info_text.set_text(f'Точек: {len(self.points)}\n'
                                   f'Треугольников: {len(triangles)}')
            
            self.fig.canvas.draw()
            
            # Показываем дополнительные визуализации
            self.triangulator.visualize()
            self.triangulator.visualize_steps()
            
        except Exception as e:
            self.info_text.set_text(f'Ошибка триангуляции:\n{str(e)}')


def demonstrate():
    """
    Демонстрация работы алгоритма триангуляции методом вторгающейся вершины.
    """
    
    print("=" * 60)
    print("ТРИАНГУЛЯЦИЯ МЕТОДОМ ВТОРГАЮЩЕЙСЯ ВЕРШИНЫ")
    print("=" * 60)
    
    print("\nВыберите режим работы:")
    print("1. Интерактивное построение полигона")
    print("2. Демонстрация на готовом примере")
    
    choice = input("\nВаш выбор (1 или 2): ").strip()
    
    if choice == "1":
        print("\n" + "="*60)
        print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("="*60)
        print("\nИнструкция:")
        print("- Левый клик мыши - добавить точку")
        print("- Правый клик мыши - замкнуть полигон")
        print("- Кнопка 'Триангулировать' - выполнить триангуляцию")
        print("- Кнопка 'Очистить' - начать заново")
        print("- Кнопка 'Демо' - загрузить пример полигона")
        print("\nЗапуск интерактивного режима...")
        
        builder = InteractivePolygonBuilder()
        builder.start()
        
    else:
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
        print("="*60)
        
        # Определяем вершины тестового полигона
        # Вершины должны быть упорядочены против часовой стрелки
        polygon_points = [
            (1, 1),     # вершина 0
            (4, 0.5),   # вершина 1
            (6, 2),     # вершина 2
            (5, 4),     # вершина 3
            (3, 3.5),   # вершина 4
            (2, 5),     # вершина 5
            (0, 3)      # вершина 6
        ]
        
        print("\nВершины полигона:")
        for i, (x, y) in enumerate(polygon_points):
            print(f"  Вершина {i}: ({x}, {y})")
        
        # Создаём экземпляр триангулятора
        triangulator = IntrudingVertexTriangulation(polygon_points)
        
        # Выполняем триангуляцию
        triangles = triangulator.triangulate()
        
        # Выводим результаты
        print(f"\nРезультат триангуляции:")
        print(f"  Количество треугольников: {len(triangles)}")
        print(f"  Ожидаемое количество: {len(polygon_points) - 2}")  # формула: n-2
        
        # Показываем, какие треугольники получились
        for i, triangle in enumerate(triangles):
            indices = [v.index for v in triangle]
            print(f"  Треугольник {i}: вершины {indices}")
        
        # Визуализация результата
        print("\nВизуализация результата...")
        triangulator.visualize()
        
        # Пошаговая визуализация процесса
        print("\nПошаговая визуализация...")
        triangulator.visualize_steps()

if __name__ == "__main__":
    demonstrate()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import math

# --- Вспомогательные геометрические функции ---

def calculate_polygon_area(polygon):
    """
    Вычисляет ориентированную площадь полигона (формула шнурков).
    Знак результата говорит о направлении обхода вершин:
    > 0 для обхода против часовой стрелки (CCW)
    < 0 для обхода по часовой стрелке (CW)
    """
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0

def is_ccw(polygon):
    """Проверяет, что полигон задан в порядке против часовой стрелки."""
    return calculate_polygon_area(polygon) > 0

def cross_product(p1, p2, p3):
    """
    Вычисляет псевдоскалярное (векторное) произведение векторов (p2-p1) и (p3-p2).
    Знак результата определяет направление поворота:
    > 0: поворот налево (вершина p2 выпуклая для CCW полигона)
    < 0: поворот направо (вершина p2 вогнутая для CCW полигона)
    = 0: точки коллинеарны
    """
    return (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

def is_point_in_triangle(pt, v1, v2, v3):
    """
    Проверяет, находится ли точка pt внутри треугольника (v1, v2, v3).
    Использует барицентрические координаты (через знаки векторного произведения).
    Точка внутри, если она находится с одной стороны от всех трех ребер.
    """
    # Для CCW треугольника точка должна быть "слева" от каждого ребра
    d1 = cross_product(v1, v2, pt)
    d2 = cross_product(v2, v3, pt)
    d3 = cross_product(v3, v1, pt)
    
    # Если все знаки одинаковы (и не 0), точка внутри.
    # Допускаем точки на границе (>= 0)
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or \
           (d1 <= 0 and d2 <= 0 and d3 <= 0)


# --- Основной алгоритм Ear Clipping ---

def triangulate_ear_clipping(polygon):
    """
    Триангулирует простой многоугольник методом отрезания "ушей".
    :param polygon: список вершин [(x1, y1), (x2, y2), ...]
    :return: список треугольников [[(x1,y1), (x2,y2), (x3,y3)], ...]
    """
    if len(polygon) < 3:
        raise ValueError("Полигон должен иметь как минимум 3 вершины.")
    if len(polygon) == 3:
        return [polygon]

    # Создаем рабочую копию, чтобы не изменять исходный список
    local_polygon = list(polygon)

    # Убедимся, что обход против часовой стрелки (CCW)
    if not is_ccw(local_polygon):
        local_polygon.reverse()
        
    triangles = []
    
    # Используем индексы для безопасной работы с изменяющимся списком
    vertex_indices = list(range(len(local_polygon)))
    
    # Безопасный счетчик, чтобы избежать бесконечного цикла при некорректных данных
    iterations = 0
    max_iterations = len(local_polygon) * 2 

    while len(vertex_indices) > 3 and iterations < max_iterations:
        iterations += 1
        found_ear = False
        for i in range(len(vertex_indices)):
            # Получаем индексы предыдущей, текущей и следующей вершин
            prev_idx_ptr = (i - 1 + len(vertex_indices)) % len(vertex_indices)
            curr_idx_ptr = i
            next_idx_ptr = (i + 1) % len(vertex_indices)

            p_prev_idx = vertex_indices[prev_idx_ptr]
            p_curr_idx = vertex_indices[curr_idx_ptr]
            p_next_idx = vertex_indices[next_idx_ptr]

            p_prev = local_polygon[p_prev_idx]
            p_curr = local_polygon[p_curr_idx]
            p_next = local_polygon[p_next_idx]

            # 1. Проверяем, является ли вершина выпуклой (левый поворот для CCW)
            if cross_product(p_prev, p_curr, p_next) < 0:
                continue # Это вогнутая вершина, не может быть "ухом"

            # 2. Проверяем, не лежат ли другие вершины внутри этого треугольника
            is_ear = True
            for j in range(len(local_polygon)):
                # Проверяем только те вершины, которые не являются вершинами "уха"
                if j not in (p_prev_idx, p_curr_idx, p_next_idx):
                    if is_point_in_triangle(local_polygon[j], p_prev, p_curr, p_next):
                        is_ear = False
                        break
            
            if is_ear:
                # Нашли "ухо"!
                triangles.append([p_prev, p_curr, p_next])
                # "Отрезаем" его, удаляя центральную вершину из списка индексов
                vertex_indices.pop(curr_idx_ptr)
                found_ear = True
                break # Начинаем поиск заново с уменьшенным полигоном
        
        if not found_ear:
             # Если за целый проход не нашли ухо, возможно, полигон не простой
             # или есть коллинеарные вершины. В простом алгоритме это может вызвать зацикливание.
             print("Не удалось найти ухо. Возможно, полигон сложный или имеет вырожденные случаи.")
             break

    # Добавляем последний оставшийся треугольник
    if len(vertex_indices) == 3:
        last_triangle = [local_polygon[i] for i in vertex_indices]
        triangles.append(last_triangle)
        
    return triangles

# --- Функция для визуализации ---

def plot_triangulation(polygon, triangles):
    """Рисует полигон и его триангуляцию."""
    fig, ax = plt.subplots()
    
    # Рисуем исходный полигон
    poly_patch = PolygonPatch(polygon, facecolor='none', edgecolor='blue', linewidth=2, label='Исходный полигон')
    ax.add_patch(poly_patch)
    
    # Рисуем каждый треугольник
    for tri in triangles:
        tri_patch = PolygonPatch(tri, facecolor='red', alpha=0.3, edgecolor='black')
        ax.add_patch(tri_patch)
        
    # Настраиваем отображение
    all_x = [p[0] for p in polygon]
    all_y = [p[1] for p in polygon]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Триангуляция методом вторгающихся вершин")
    plt.grid(True)
    plt.show()

# --- Пример использования ---

if __name__ == '__main__':
    # Пример невыпуклого полигона (похож на звезду или стрелку)
    # Задан в порядке обхода по часовой стрелке (алгоритм сам это определит и исправит)
    my_polygon = [
        (0, 0), (4, -1), (5, 3), (2, 2), (1, 5), (-2, 3)
    ]

    print("Исходный полигон:", my_polygon)

    # Выполняем триангуляцию
    try:
        result_triangles = triangulate_ear_clipping(my_polygon)
        print(f"\nНайдено {len(result_triangles)} треугольников:")
        for i, t in enumerate(result_triangles):
            print(f"Треугольник {i+1}: {t}")

        # Визуализируем результат
        plot_triangulation(my_polygon, result_triangles)
        
    except ValueError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
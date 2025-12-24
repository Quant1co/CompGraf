import random
import matplotlib.pyplot as plt

def get_position(a, b, p):
    """
    Определяет положение точки P относительно вектора AB.
    Используется косое произведение (Cross Product).
    
    Возвращает:
    > 0 : P слева от AB
    < 0 : P справа от AB
    = 0 : P на прямой AB (коллинеарна)
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

def get_dist(a, b, p):
    """
    Вычисляет расстояние (пропорциональное) от точки P до прямой AB.
    Нам не нужно делить на длину отрезка, так как мы просто сравниваем расстояния.
    Используем абсолютное значение косого произведения.
    """
    return abs(get_position(a, b, p))

def find_hull(points, p1, p2):
    """
    Рекурсивная функция для поиска точек оболочки справа от вектора p1->p2.
    """
    if not points:
        return []

    # 1. Находим точку, наиболее удаленную от отрезка p1-p2
    farthest_point = None
    max_dist = -1

    for p in points:
        d = get_dist(p1, p2, p)
        if d > max_dist:
            max_dist = d
            farthest_point = p

    # 2. Эта точка точно входит в оболочку
    # Теперь нужно найти точки, лежащие снаружи треугольника p1-farthest-p2
    
    # Подмножество точек слева от вектора p1 -> farthest
    set1 = [p for p in points if get_position(p1, farthest_point, p) > 0]
    
    # Подмножество точек слева от вектора farthest -> p2
    set2 = [p for p in points if get_position(farthest_point, p2, p) > 0]

    # 3. Рекурсивно запускаем поиск для двух новых граней
    # Обратите внимание: точки внутри треугольника уже отсеялись,
    # так как они будут справа от обоих векторов.
    hull_part1 = find_hull(set1, p1, farthest_point)
    hull_part2 = find_hull(set2, farthest_point, p2)

    # Собираем результат: левая часть + вершина + правая часть
    return hull_part1 + [farthest_point] + hull_part2

def quick_hull(points):
    """
    Основная функция алгоритма QuickHull.
    """
    if len(points) < 3:
        return points

    # 1. Находим самую левую (min_x) и самую правую (max_x) точки.
    # Они гарантированно входят в выпуклую оболочку.
    min_x = min(points, key=lambda p: p[0])
    max_x = max(points, key=lambda p: p[0])

    # 2. Делим остальные точки на две группы относительно линии min_x -> max_x
    # left_set: точки слева от вектора (верхняя часть)
    # right_set: точки справа от вектора (нижняя часть), 
    # но для удобства мы будем искать "слева" от обратного вектора max_x -> min_x
    
    left_set = [p for p in points if get_position(min_x, max_x, p) > 0]
    right_set = [p for p in points if get_position(max_x, min_x, p) > 0]

    # 3. Запускаем рекурсию для верхней и нижней части
    upper_hull = find_hull(left_set, min_x, max_x)
    lower_hull = find_hull(right_set, max_x, min_x)

    # 4. Собираем итоговый список вершин (против часовой стрелки)
    return [min_x] + upper_hull + [max_x] + lower_hull

# --- Блок визуализации и проверки ---

def visualize(points, hull_points):
    x_pts, y_pts = zip(*points)
    
    # Чтобы замкнуть контур на графике, добавим первую точку в конец
    hull_to_plot = hull_points + [hull_points[0]]
    hx, hy = zip(*hull_to_plot)

    plt.figure(figsize=(10, 6))
    plt.title("Convex Hull (QuickHull Algorithm)")
    
    # Рисуем все точки
    plt.scatter(x_pts, y_pts, color='blue', s=10, label='Исходные точки')
    
    # Рисуем оболочку
    plt.plot(hx, hy, color='red', linewidth=2, marker='o', label='Выпуклая оболочка')
    
    # Закрасим область внутри
    plt.fill(hx, hy, 'red', alpha=0.1)
    
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Генерируем 50 случайных точек
    N = 50
    random_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(N)]
    
    # Удаляем дубликаты, если есть (алгоритм не любит совпадающие точки)
    random_points = list(set(random_points))

    # Вычисляем оболочку
    hull = quick_hull(random_points)

    print(f"Количество исходных точек: {len(random_points)}")
    print(f"Точек в выпуклой оболочке: {len(hull)}")
    
    # Рисуем
    visualize(random_points, hull)
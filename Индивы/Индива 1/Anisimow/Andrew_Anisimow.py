import matplotlib.pyplot as plt
import numpy as np

# Глобальные переменные
points = []  # Список точек, добавляемых пользователем
ax = None    # Ось для графика
fig = None   # Фигура

# Функция для вычисления кросс-произведения (ориентация трёх точек)
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

# Функция для обработки клика мыши (добавление точки)
def on_click(event):
    if event.button == 1:  # Левая кнопка мыши
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            points.append((x, y))
            ax.scatter(x, y, color='blue')
            plt.draw()  # Обновить график

# Функция для обработки нажатия клавиши (пробел запускает алгоритм)
def on_key(event):
    if event.key == ' ' and len(points) >= 3:
        animate_convex_hull()

# Пошаговая анимация построения выпуклой оболочки
def animate_convex_hull():
    global points
    # Сортировка точек по x, затем по y
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    
    # Отметить крайнюю левую и правую точки
    left_point = sorted_points[0]
    right_point = sorted_points[-1]
    ax.scatter(left_point[0], left_point[1], color='purple', s=100, label='Крайняя левая', zorder=10)
    ax.scatter(right_point[0], right_point[1], color='magenta', s=100, label='Крайняя правая', zorder=10)
    plt.draw()
    plt.pause(1)  # Пауза, чтобы показать отмеченные точки
    
    # Нарисовать тонкую жёлтую линию между левой и правой
    yellow_line, = ax.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]], 
                           color='yellow', linewidth=1, label='Разделяющая линия')
    plt.draw()
    plt.pause(0.5)  # Пауза, чтобы показать линию
    
    # Очистка предыдущих линий (кроме жёлтой)
    for line in ax.get_lines():
        if line.get_label() != 'Разделяющая линия':
            line.remove()
    
    # Построение нижней оболочки с анимацией
    lower = []
    lower_line, = ax.plot([], [], color='green', marker='o', label='Нижняя оболочка')  # Линия для нижней
    for p in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
            # Обновить линию после удаления
            lower_line.set_data([pt[0] for pt in lower], [pt[1] for pt in lower])
            plt.draw()
            plt.pause(0.5)  # Пауза для визуализации
        lower.append(p)
        # Обновить линию
        lower_line.set_data([pt[0] for pt in lower], [pt[1] for pt in lower])
        plt.draw()
        plt.pause(0.5)  # Пауза для визуализации
    
    # Построение верхней оболочки с анимацией (в обратном порядке)
    upper = []
    upper_line, = ax.plot([], [], color='orange', marker='o', label='Верхняя оболочка')  # Линия для верхней
    for p in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
            # Обновить линию после удаления
            upper_line.set_data([pt[0] for pt in upper], [pt[1] for pt in upper])
            plt.draw()
            plt.pause(0.5)
        upper.append(p)
        # Обновить линию
        upper_line.set_data([pt[0] for pt in upper], [pt[1] for pt in upper])
        plt.draw()
        plt.pause(0.5)
    
    # Объединение в финальную оболочку
    hull = lower[:-1] + upper[:-1]
    hull_np = np.array(hull)
    
    # Замыкание полигона: добавить первую точку в конец для полной линии
    hull_np_closed = np.vstack([hull_np, hull_np[0]])
    
    # Удалить промежуточные линии (жёлтая остаётся)
    lower_line.remove()
    upper_line.remove()
    
    # Нарисовать финальную оболочку с жирной линией
    ax.plot(hull_np_closed[:, 0], hull_np_closed[:, 1], color='red', marker='o', linewidth=2, label='Выпуклая оболочка')
    ax.fill(hull_np[:, 0], hull_np[:, 1], alpha=0.2, color='red')  # Заливка без замыкания в plot, но fill работает
    plt.legend()
    plt.draw()

# Основная функция для запуска
def main():
    global ax, fig
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title('Кликните для добавления точек. Пробел - построить оболочку.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    
    # Подключение событий
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()

if __name__ == "__main__":
    main()
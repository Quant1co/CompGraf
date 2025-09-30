from PIL import Image, ImageDraw
import math

def bresenham_line(draw, x0, y0, x1, y1, color):
    """
    Рисует отрезок с помощью целочисленного алгоритма Брезенхема.
    Работает для всех направлений и наклонов.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # Определяем направление движения по осям
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Переменная ошибки
    err = dx - dy

    while True:
        draw.point((x0, y0), fill=color)

        # Если достигли конечной точки, выходим
        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        # Корректируем ошибку и делаем шаг по X
        if e2 > -dy:
            err -= dy
            x0 += sx

        # Корректируем ошибку и делаем шаг по Y
        if e2 < dx:
            err += dx
            y0 += sy


# Вспомогательные функции для алгоритма Ву
def ipart(x):
    return math.floor(x)


def fpart(x):
    return x - math.floor(x)


def rfpart(x):
    return 1 - fpart(x)


def wu_line(draw, x0, y0, x1, y1, color_rgb):
    """
    Рисует отрезок с помощью алгоритма Ву (с сглаживанием).
    """

    def plot(x, y, intensity):
        """ Вспомогательная функция для отрисовки пикселя с заданной интенсивностью """
        # Интенсивность (0-1) преобразуется в альфа-канал (0-255)
        alpha = int(255 * intensity)
        # Создаем цвет с учетом альфа-канала
        color_with_alpha = color_rgb + (alpha,)
        draw.point((x, y), fill=color_with_alpha)

    dx = x1 - x0
    dy = y1 - y0

    # Определяем, является ли линия "крутой" (изменение по Y больше, чем по X)
    steep = abs(dy) > abs(dx)

    if steep:
        # Если линия крутая, меняем местами X и Y
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dx, dy = dy, dx

    if x0 > x1:
        # Убеждаемся, что мы всегда рисуем слева направо
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0

    # Вычисляем градиент
    gradient = dy / dx if dx != 0 else 1.0

    # Обработка начальной точки
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = ipart(yend)

    if steep:
        plot(ypxl1, xpxl1, rfpart(yend) * xgap)
        plot(ypxl1 + 1, xpxl1, fpart(yend) * xgap)
    else:
        plot(xpxl1, ypxl1, rfpart(yend) * xgap)
        plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap)

    intery = yend + gradient

    # Обработка конечной точки
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = fpart(x1 + 0.5)
    xpxl2 = xend
    ypxl2 = ipart(yend)

    if steep:
        plot(ypxl2, xpxl2, rfpart(yend) * xgap)
        plot(ypxl2 + 1, xpxl2, fpart(yend) * xgap)
    else:
        plot(xpxl2, ypxl2, rfpart(yend) * xgap)
        plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap)

    # Основной цикл - итерируем по X между начальной и конечной точками
    for x in range(int(xpxl1 + 1), int(xpxl2)):
        if steep:
            plot(ipart(intery), x, rfpart(intery))
            plot(ipart(intery) + 1, x, fpart(intery))
        else:
            plot(x, ipart(intery), rfpart(intery))
            plot(x, ipart(intery) + 1, fpart(intery))
        intery += gradient


if __name__ == '__main__':
    width, height = 400, 400

    # --- Создаем изображение для Брезенхема ---
    img_bresenham = Image.new('RGB', (width, height), 'black')
    draw_bresenham = ImageDraw.Draw(img_bresenham)

    # Рисуем несколько линий
    bresenham_line(draw_bresenham, 20, 30, 380, 150, 'red')
    bresenham_line(draw_bresenham, 380, 180, 20, 250, 'green')
    bresenham_line(draw_bresenham, 50, 380, 150, 20, 'blue')

    img_bresenham.save('bresenham_lines.png')
    print("Изображение 'bresenham_lines.png' сохранено.")

    # --- Создаем изображение для Ву ---
    # Используем RGBA, так как Ву работает с прозрачностью (альфа-канал)
    # Фон должен быть непрозрачным
    img_wu = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    draw_wu = ImageDraw.Draw(img_wu)

    # Рисуем те же линии
    wu_line(draw_wu, 20, 30, 380, 150, (255, 0, 0))  # Red
    wu_line(draw_wu, 380, 180, 20, 250, (0, 255, 0))  # Green
    wu_line(draw_wu, 50, 380, 150, 20, (0, 0, 255))  # Blue

    # Для правильного отображения нужно скомпоновать с фоном
    # Создаем черный фон и накладываем на него наше изображение с альфа-каналом
    final_img_wu = Image.new('RGB', (width, height), 'black')
    final_img_wu.paste(img_wu, (0, 0), img_wu)

    final_img_wu.save('wu_lines.png')
    print("Изображение 'wu_lines.png' сохранено.")
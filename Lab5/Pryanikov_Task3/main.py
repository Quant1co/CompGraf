import pygame
import numpy as np
from typing import List, Tuple, Optional
import math

class BezierPoint:
    """Класс для представления опорной точки"""
    def __init__(self, x: float, y: float, point_type: str = 'anchor'):
        self.x = x
        self.y = y
        self.type = point_type  # 'anchor' или 'control'
        self.selected = False
        
    def get_pos(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def set_pos(self, x: float, y: float):
        self.x = x
        self.y = y
        
    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)

class CubicBezierSpline:
    """Класс для составной кубической кривой Безье"""
    def __init__(self):
        self.points: List[BezierPoint] = []
        self.dragging_point: Optional[BezierPoint] = None
        self.show_control_points = True
        self.show_control_lines = True
        
    def add_point(self, x: float, y: float):
        """Добавляет новую опорную точку с контрольными точками"""
        if len(self.points) == 0:
            # Первая точка
            self.points.append(BezierPoint(x, y, 'anchor'))
        else:
            # Добавляем новый сегмент
            last_anchor = self.get_last_anchor()
            if last_anchor:
                # Добавляем две контрольные точки и новую опорную
                ctrl1_x = last_anchor.x + (x - last_anchor.x) * 0.33
                ctrl1_y = last_anchor.y + (y - last_anchor.y) * 0.33
                ctrl2_x = last_anchor.x + (x - last_anchor.x) * 0.66
                ctrl2_y = last_anchor.y + (y - last_anchor.y) * 0.66
                
                self.points.append(BezierPoint(ctrl1_x, ctrl1_y, 'control'))
                self.points.append(BezierPoint(ctrl2_x, ctrl2_y, 'control'))
                self.points.append(BezierPoint(x, y, 'anchor'))
    
    def get_last_anchor(self) -> Optional[BezierPoint]:
        """Возвращает последнюю опорную точку"""
        for point in reversed(self.points):
            if point.type == 'anchor':
                return point
        return None
    
    def remove_point(self, x: float, y: float, threshold: float = 10):
        """Удаляет ближайшую точку"""
        if len(self.points) <= 1:
            return
            
        closest_point = None
        min_dist = float('inf')
        
        for point in self.points:
            dist = point.distance_to(x, y)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_point = point
        
        if closest_point:
            idx = self.points.index(closest_point)
            
            if closest_point.type == 'anchor':
                # Удаляем опорную точку и связанные контрольные точки
                if idx == 0:  # Первая опорная точка
                    if len(self.points) > 3:
                        del self.points[0:4]  # Удаляем точку и следующий сегмент
                    else:
                        self.points.clear()
                elif idx == len(self.points) - 1:  # Последняя опорная точка
                    del self.points[idx-2:idx+1]  # Удаляем точку и предыдущие контрольные
                else:
                    # Средняя опорная точка - объединяем сегменты
                    del self.points[idx-2:idx+3]
            else:
                # Просто удаляем контрольную точку
                del self.points[idx]
    
    def get_point_at(self, x: float, y: float, threshold: float = 10) -> Optional[BezierPoint]:
        """Находит точку рядом с указанными координатами"""
        for point in self.points:
            if point.distance_to(x, y) < threshold:
                return point
        return None
    
    def cubic_bezier(self, p0: Tuple[float, float], p1: Tuple[float, float],
                     p2: Tuple[float, float], p3: Tuple[float, float],
                     t: float) -> Tuple[float, float]:
        """Вычисляет точку на кубической кривой Безье"""
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        
        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        
        return (x, y)
    
    def get_curve_points(self, num_points: int = 100) -> List[Tuple[float, float]]:
        """Генерирует точки для отрисовки кривой"""
        curve_points = []
        
        # Находим все сегменты (группы по 4 точки: anchor, control, control, anchor)
        anchors = [p for p in self.points if p.type == 'anchor']
        
        if len(anchors) < 2:
            return curve_points
        
        # Для каждого сегмента
        i = 0
        while i < len(self.points) - 3:
            if self.points[i].type == 'anchor':
                # Нашли начало сегмента
                p0 = self.points[i].get_pos()
                p1 = self.points[i + 1].get_pos()
                p2 = self.points[i + 2].get_pos()
                p3 = self.points[i + 3].get_pos()
                
                # Генерируем точки кривой
                for j in range(num_points):
                    t = j / (num_points - 1)
                    point = self.cubic_bezier(p0, p1, p2, p3, t)
                    curve_points.append(point)
                
                i += 3  # Переходим к следующему сегменту
            else:
                i += 1
        
        return curve_points

class BezierEditor:
    """Главный класс редактора"""
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Редактор кубических сплайнов Безье")
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.spline = CubicBezierSpline()
        self.font = pygame.font.Font(None, 24)
        
        # Цвета
        self.bg_color = (30, 30, 40)
        self.curve_color = (100, 200, 255)
        self.anchor_color = (255, 100, 100)
        self.control_color = (100, 255, 100)
        self.control_line_color = (80, 80, 80)
        self.selected_color = (255, 255, 100)
        self.grid_color = (50, 50, 60)
        self.text_color = (200, 200, 200)
    
    def draw_grid(self):
        """Рисует сетку"""
        step = 50
        for x in range(0, self.width, step):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, step):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.width, y), 1)
    
    def draw_spline(self):
        """Рисует кривую Безье"""
        curve_points = self.spline.get_curve_points()
        
        if len(curve_points) > 1:
            # Рисуем саму кривую
            pygame.draw.lines(self.screen, self.curve_color, False, curve_points, 3)
        
        # Рисуем контрольные линии
        if self.spline.show_control_lines:
            i = 0
            while i < len(self.spline.points) - 3:
                if self.spline.points[i].type == 'anchor':
                    # Линии от опорных точек к контрольным
                    pygame.draw.line(self.screen, self.control_line_color,
                                   self.spline.points[i].get_pos(),
                                   self.spline.points[i + 1].get_pos(), 1)
                    pygame.draw.line(self.screen, self.control_line_color,
                                   self.spline.points[i + 2].get_pos(),
                                   self.spline.points[i + 3].get_pos(), 1)
                    i += 3
                else:
                    i += 1
        
        # Рисуем точки
        for point in self.spline.points:
            if point.type == 'anchor':
                color = self.selected_color if point.selected else self.anchor_color
                pygame.draw.circle(self.screen, color, 
                                 (int(point.x), int(point.y)), 8)
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(point.x), int(point.y)), 8, 2)
            elif self.spline.show_control_points:
                color = self.selected_color if point.selected else self.control_color
                pygame.draw.circle(self.screen, color, 
                                 (int(point.x), int(point.y)), 6)
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (int(point.x), int(point.y)), 6, 1)
    
    def draw_ui(self):
        """Рисует интерфейс"""
        instructions = [
            "Управление:",
            "ЛКМ - добавить точку / переместить точку",
            "ПКМ - удалить точку",
            "C - показать/скрыть контрольные точки",
            "L - показать/скрыть контрольные линии",
            "R - очистить все",
            "ESC - выход"
        ]
        
        y_offset = 10
        for text in instructions:
            rendered = self.font.render(text, True, self.text_color)
            self.screen.blit(rendered, (10, y_offset))
            y_offset += 25
        
        # Показываем количество точек
        num_anchors = len([p for p in self.spline.points if p.type == 'anchor'])
        info_text = f"Опорных точек: {num_anchors}"
        rendered = self.font.render(info_text, True, self.text_color)
        self.screen.blit(rendered, (10, self.height - 30))
    
    def handle_events(self):
        """Обработка событий"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_c:
                    self.spline.show_control_points = not self.spline.show_control_points
                elif event.key == pygame.K_l:
                    self.spline.show_control_lines = not self.spline.show_control_lines
                elif event.key == pygame.K_r:
                    self.spline.points.clear()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                
                if event.button == 1:  # ЛКМ
                    # Проверяем, есть ли точка рядом
                    point = self.spline.get_point_at(x, y)
                    if point:
                        # Начинаем перетаскивание
                        self.spline.dragging_point = point
                        point.selected = True
                    else:
                        # Добавляем новую точку
                        self.spline.add_point(x, y)
                
                elif event.button == 3:  # ПКМ
                    # Удаляем точку
                    self.spline.remove_point(x, y)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.spline.dragging_point:
                    self.spline.dragging_point.selected = False
                    self.spline.dragging_point = None
            
            elif event.type == pygame.MOUSEMOTION:
                if self.spline.dragging_point:
                    x, y = pygame.mouse.get_pos()
                    self.spline.dragging_point.set_pos(x, y)
    
    def run(self):
        """Главный цикл программы"""
        while self.running:
            self.handle_events()
            
            # Очистка экрана
            self.screen.fill(self.bg_color)
            
            # Рисование
            self.draw_grid()
            self.draw_spline()
            self.draw_ui()
            
            # Обновление экрана
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    editor = BezierEditor()
    editor.run()
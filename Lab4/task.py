import pygame
import sys

# Класс для полигона
class Polygon:
    def __init__(self):
        self.vertices = []  # Список вершин (точек) полигона

    def add_vertex(self, point):
        self.vertices.append(point)

    def draw(self, screen, color=(255, 0, 0)):
        if len(self.vertices) == 1:
            # Отрисовка точки
            pygame.draw.circle(screen, color, self.vertices[0], 3)
        elif len(self.vertices) >= 2:
            # Отрисовка линий между вершинами
            pygame.draw.lines(screen, color, True, self.vertices, 2)  # True для замкнутого полигона

# Основной класс приложения
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Lab 4: Affine Transformations - Part 1")
        self.clock = pygame.time.Clock()
        self.polygons = []  # Список всех полигонов на сцене
        self.current_polygon = None  # Текущий создаваемый полигон
        self.running = True

    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  
                    pos = pygame.mouse.get_pos()
                    if self.current_polygon is None:
                        self.current_polygon = Polygon()
                    self.current_polygon.add_vertex(pos)
                    print(f"Added vertex: {pos}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  
                    if self.current_polygon and len(self.current_polygon.vertices) > 0:
                        self.polygons.append(self.current_polygon)
                        self.current_polygon = None
                        print("Polygon completed")
                elif event.key == pygame.K_r:  
                    self.polygons = []
                    self.current_polygon = None
                    print("Scene cleared")
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def draw(self):
        self.screen.fill((255, 255, 255))  # фон пусть будет белый

        # Отрисовка всех полигонов
        for poly in self.polygons:
            poly.draw(self.screen)

        # Отрисовка текущего полигона, если он создается
        if self.current_polygon:
            self.current_polygon.draw(self.screen, color=(0, 255, 0))  # полигон пока выбран горит зелёным

        pygame.display.flip()

if __name__ == "__main__":
    app = App()
    app.run()
    pygame.quit()
    sys.exit()

    # это и что ниже можно удалить
    # очистить сцену на R
    # пробел - завершить рисование полигона
    # ЛКМ добавить вершину текущему полигону
import cv2
import numpy as np

# --- НАСТРОЙКИ ---
IMAGE_PATH = 'image.jpg'  # <-- Укажите путь к вашему изображению
OUTPUT_PATH = 'result.jpg'  # <-- Имя файла для сохранения результата
WINDOW_NAME_IMAGE = 'Image'
WINDOW_NAME_CONTROLS = 'Controls'

# Максимальный размер окна для отображения картинки (чтобы помещалась на экране)
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900

# Настройки панели управления
PANEL_HEIGHT = 180
PANEL_WIDTH = 400
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 150
BUTTON_COLOR = (80, 80, 80)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Глобальная переменная для передачи действия от кнопки в главный цикл
last_button_action = None


def nothing(x):
    """Пустая функция-заглушка для createTrackbar"""
    pass


def mouse_callback(event, x, y, flags, param):
    """Обработчик кликов мыши по окну с элементами управления."""
    global last_button_action

    if event == cv2.EVENT_LBUTTONDOWN:
        # Проверка клика по кнопке "Сохранить"
        if btn_save_y < y < btn_save_y + BUTTON_HEIGHT and btn_save_x < x < btn_save_x + BUTTON_WIDTH:
            print("Нажата кнопка 'Сохранить'")
            last_button_action = 'save'

        # Проверка клика по кнопке "Выход"
        if btn_quit_y < y < btn_quit_y + BUTTON_HEIGHT and btn_quit_x < x < btn_quit_x + BUTTON_WIDTH:
            print("Нажата кнопка 'Выход'")
            last_button_action = 'quit'


# Загружаем изображение
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Ошибка: не удалось загрузить изображение по пути: {IMAGE_PATH}")
    exit()

# --- Масштабируем изображение для отображения, если оно слишком большое ---
h, w = image.shape[:2]
if w > MAX_DISPLAY_WIDTH or h > MAX_DISPLAY_HEIGHT:
    ratio = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h)
    new_dim = (int(w * ratio), int(h * ratio))
    image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    print(f"Изображение было уменьшено до {new_dim[0]}x{new_dim[1]} для удобства отображения.")

# Запоминаем оригинальное состояние для сброса изменений
hsv_original = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Создание окон
cv2.namedWindow(WINDOW_NAME_IMAGE)
cv2.namedWindow(WINDOW_NAME_CONTROLS)
cv2.setMouseCallback(WINDOW_NAME_CONTROLS, mouse_callback)

# Создание ползунков (Trackbars)
cv2.createTrackbar('H Shift', WINDOW_NAME_CONTROLS, 0, 179, nothing)
cv2.createTrackbar('S Add', WINDOW_NAME_CONTROLS, 255, 510, nothing)
cv2.createTrackbar('V Add', WINDOW_NAME_CONTROLS, 255, 510, nothing)

print("Инструкция:")
print(" - Двигайте ползунки для изменения цвета (H), насыщенности (S) и яркости (V).")
print(" - Нажмите на кнопку 'Save' для сохранения результата.")
print(" - Нажмите на кнопку 'Quit' или клавишу 'q'/ESC для выхода.")

while True:
    # Создаем и отрисовываем панель управления в каждом кадре
    control_panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (50, 50, 50), dtype=np.uint8)

    # Координаты кнопок (пересчитываются в цикле для простоты)
    btn_save_x = int((PANEL_WIDTH - BUTTON_WIDTH) / 2)
    btn_save_y = 90
    btn_quit_x = int((PANEL_WIDTH - BUTTON_WIDTH) / 2)
    btn_quit_y = 135

    cv2.rectangle(control_panel, (btn_save_x, btn_save_y), (btn_save_x + BUTTON_WIDTH, btn_save_y + BUTTON_HEIGHT),
                  BUTTON_COLOR, -1)
    cv2.putText(control_panel, 'Save', (btn_save_x + 50, btn_save_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                BUTTON_TEXT_COLOR, 2)

    cv2.rectangle(control_panel, (btn_quit_x, btn_quit_y), (btn_quit_x + BUTTON_WIDTH, btn_quit_y + BUTTON_HEIGHT),
                  BUTTON_COLOR, -1)
    cv2.putText(control_panel, 'Quit', (btn_quit_x + 50, btn_quit_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                BUTTON_TEXT_COLOR, 2)

    cv2.imshow(WINDOW_NAME_CONTROLS, control_panel)

    # --- Основная логика обработки изображения ---
    hsv_modified = hsv_original.copy()

    h_shift = cv2.getTrackbarPos('H Shift', WINDOW_NAME_CONTROLS)
    s_add = cv2.getTrackbarPos('S Add', WINDOW_NAME_CONTROLS) - 255
    v_add = cv2.getTrackbarPos('V Add', WINDOW_NAME_CONTROLS) - 255

    h, s, v = cv2.split(hsv_modified)

    # Применяем изменения с корректным переполнением и ограничением
    h = ((h.astype(np.int16) + h_shift) % 180).astype(np.uint8)
    s = np.clip(s.astype(np.int16) + s_add, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.int16) + v_add, 0, 255).astype(np.uint8)

    hsv_modified = cv2.merge([h, s, v])
    result_bgr = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    cv2.imshow(WINDOW_NAME_IMAGE, result_bgr)

    # --- Обработка действий от кнопок ---
    if last_button_action == 'save':
        cv2.imwrite(OUTPUT_PATH, result_bgr)
        print(f"Изображение сохранено в файл: {OUTPUT_PATH}")
        last_button_action = None  # Сбрасываем действие, чтобы не сохранять в каждом кадре

    if last_button_action == 'quit':
        break

        # Обработка нажатия клавиш (альтернативный способ выхода)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()
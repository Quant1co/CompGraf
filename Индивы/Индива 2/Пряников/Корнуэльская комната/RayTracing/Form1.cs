using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace RayTracing
{
    public partial class Form1 : Form
    {
        public List<Figure> scene = new List<Figure>();   // список фигур сцены (комната + объекты)
        public List<Light> lights = new List<Light>();    // список источников света
        public Color[,] color_pixels;                     // цвета пикселей для отображения на pictureBox
        public Point3D[,] pixels;                         // координаты точек экранной плоскости (на передней стене)
        public Point3D focus;                              // точка наблюдателя (камера)
        public Point3D up_left, up_right, down_left, down_right; // углы экранной плоскости
        public int h, w;                                   // высота/ширина области рендера

        public Form1()
        {
            InitializeComponent();
            // Инициализация базовых параметров камеры и плоскости
            focus = new Point3D();
            up_left = new Point3D();
            up_right = new Point3D();
            down_left = new Point3D();
            down_right = new Point3D();
            h = pictureBox1.Height;
            w = pictureBox1.Width;
            pictureBox1.Image = new Bitmap(w, h);
            // Начальные состояния чекбоксов материалов/света
            cubeSpecularCB.Checked = false;             // прозрачный куб
            sphereSpecularCB.Checked = false;           // прозрачный шар
            refractCubeCB.Checked = false;              // зеркальный куб
            refractSphereCB.Checked = false;            // зеркальный шар
            frontWallSpecularCB.Checked = backWallSpecularCB.Checked = leftWallSpecularCB.Checked = rightWallSpecularCB.Checked = false; // зеркальность стен
            twoLightsCB.Checked = false;                // 2 источника света
        }

        // Построение сцены: комната, объекты, источники света, материалы
        public void build_scene()
        {
            // Создаём комнату как гексаэдр (куб)
            Figure room = Figure.GetHexahedron(10);
            // Передняя грань комнаты задаёт экранную плоскость (точки по часовой стрелке)
            up_left = room.sides[0].get_point(0);
            up_right = room.sides[0].get_point(1);
            down_right = room.sides[0].get_point(2);
            down_left = room.sides[0].get_point(3);

            // Вычисляем нормаль и центр передней грани, камера находится перед ней вдоль нормали
            Point3D normal = Side.norm(room.sides[0]);                            // нормаль стороны комнаты
            Point3D center = (up_left + up_right + down_left + down_right) / 4;   // центр стороны комнаты
            focus = center + normal * 10;                                         // позиция камеры

            room.set_pen(new Pen(Color.White));
            room.isRoom = true; // помечаем как комнату, чтобы материалы граней выбирались по индексу

            float refl, refr, amb, dif, env;

            // Задняя (передняя по коду — грань с индексом 0) стена
            room.sides[0].drawing_pen = new Pen(Color.White);
            if (backWallSpecularCB.Checked)
            {
                // Зеркальная: отражение, без диффуза/амбиента
                refl = 0.8f; refr = 0f; amb = 0.0f; dif = 0.0f; env = 1f;
            }
            else
            {
                // Диффузная: амбиент + диффуз
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.8f; env = 1f;
            }
            room.back_wall_material = new Material(refl, refr, amb, dif, env);

            // Передняя белая стена (индекс 1)
            room.sides[1].drawing_pen = new Pen(Color.White);
            if (frontWallSpecularCB.Checked)
            {
                refl = 0.8f; refr = 0f; amb = 0.0f; dif = 0.0f; env = 1f;
            }
            else
            {
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.8f; env = 1f;
            }
            room.front_wall_material = new Material(refl, refr, amb, dif, env);

            // Правая зелёная стена (классическая Корнеллская комната)
            room.sides[2].drawing_pen = new Pen(Color.Green);
            if (rightWallSpecularCB.Checked)
            {
                refl = 0.8f; refr = 0f; amb = 0.0f; dif = 0.0f; env = 1f;
            }
            else
            {
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.8f; env = 1f;
            }
            room.right_wall_material = new Material(refl, refr, amb, dif, env);

            // Левая красная стена (классическая Корнеллская комната)
            room.sides[3].drawing_pen = new Pen(Color.Red);
            if (leftWallSpecularCB.Checked)
            {
                refl = 0.8f; refr = 0f; amb = 0.0f; dif = 0.0f; env = 1f;
            }
            else
            {
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.8f; env = 1f;
            }
            room.left_wall_material = new Material(refl, refr, amb, dif, env);

            // Верхняя стена (индекс 4)
            if (upWallSpecularCB.Checked)
            {
                refl = 0.8f; refr = 0f; amb = 0.0f; dif = 0.0f; env = 1f;
            }
            else
            {
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.8f; env = 1f;
            }
            room.up_wall_material = new Material(refl, refr, amb, dif, env);

            // Нижняя стена (индекс 5)
            if (downWallSpecularCB.Checked)
            {
                refl = 0.8f; refr = 0f; amb = 0.0f; dif = 0.0f; env = 1f;
            }
            else
            {
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.8f; env = 1f;
            }
            room.down_wall_material = new Material(refl, refr, amb, dif, env);

            // Источники света: один сверху спереди, второй опционально на правой стене
            Light l1 = new Light(new Point3D(0f, 1f, 4.9f), new Point3D(1f, 1f, 1f));
            lights.Add(l1);
            if (twoLightsCB.Checked)
            {
                // Второй источник на правой стене: ближе к середине, немного выше
                Light l2 = new Light(new Point3D(4.5f, 2.0f, -3.0f), new Point3D(1f, 1f, 1f));
                lights.Add(l2);
            }

            // Куб 1: позиционируем и задаём материал (либо преломляющий, либо диффузный)
            Figure cube1 = Figure.GetHexahedron(3.2f);
            cube1.offset(-0.5f, -1, -3.5f);
            cube1.rotate_around(55, "CZ");
            cube1.set_pen(new Pen(Color.OrangeRed));
            if (refractCubeCB.Checked) // зеркальный (на самом деле преломляющий в терминах материалов)
            {
                refl = 0.0f; refr = 0.8f; amb = 0f; dif = 0.0f; env = 1.03f;
            }
            else
            {
                refl = 0f; refr = 0f; amb = 0.1f; dif = 0.7f; env = 1f;
            }
            cube1.figure_material = new Material(refl, refr, amb, dif, env);

            // Куб 2: позиционируем и задаём материал (либо зеркальный, либо диффузный)
            Figure cube2 = Figure.GetHexahedron(2.6f);
            cube2.offset(-2.4f, 2, -3.8f);
            cube2.rotate_around(30, "CZ");
            cube2.set_pen(new Pen(Color.MediumPurple));
            if (cubeSpecularCB.Checked) // прозрачный (здесь используется отражение как «зеркальность»)
            {
                refl = 0.8f; refr = 0f; amb = 0.05f; dif = 0.0f; env = 1f;
            }
            else
            {
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.8f; env = 1f;
            }
            cube2.figure_material = new Material(refl, refr, amb, dif, env);

            // Шар 1: материал преломляющий либо диффузный
            Sphere s1 = new Sphere(new Point3D(2.5f, 2f, -3.4f), 1.7f);
            s1.set_pen(new Pen(Color.DeepSkyBlue));
            if (refractSphereCB.Checked) // зеркальный (преломляющий)
            {
                refl = 0.0f; refr = 0.9f; amb = 0f; dif = 0.0f; env = 1.03f;
            }
            else
            {
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.9f; env = 1f;
            }
            s1.figure_material = new Material(refl, refr, amb, dif, env);

            // Шар 2: материал преломляющий либо диффузный
            Sphere s2 = new Sphere(new Point3D(-2.2f, 1.6f, -1.4f), 1.2f);
            s2.set_pen(new Pen(Color.LimeGreen));
            if (sphereSpecularCB.Checked) // прозрачный (преломляющий)
            {
                refl = 0.0f; refr = 0.9f; amb = 0f; dif = 0.0f; env = 1.03f;
            }
            else
            {
                refl = 0.0f; refr = 0f; amb = 0.1f; dif = 0.9f; env = 1f;
            }
            s2.figure_material = new Material(refl, refr, amb, dif, env);

            // Добавляем фигуры в сцену (комната первой)
            scene.Add(room);
            scene.Add(cube1);
            scene.Add(cube2);
            scene.Add(s2);
            scene.Add(s1);
        }

        // Очистка сцены и источников света
        public void Clear()
        {
            scene.Clear();
            lights.Clear();
        }

        // Обработчик кнопки рендера: сборка сцены, трассировка, вывод на картинку и обновление статуса
        private void button1_Click(object sender, EventArgs e)
        {
            Clear();
            build_scene();
            run_rayTrace();
            // Перенос рассчитанных цветов в Bitmap (медленно, но просто)
            for (int i = 0; i < w; ++i)
            {
                for (int j = 0; j < h; ++j)
                    (pictureBox1.Image as Bitmap).SetPixel(i, j, color_pixels[i, j]);
            }
            pictureBox1.Invalidate();
            // Обновление статуса рендера
            int figuresCount = scene.Count - 1; // без комнаты
            int spheresCount = scene.Count(f => f is Sphere);
            int cubesCount = figuresCount - spheresCount;
            int lightsCount = lights.Count;
            string specWalls = string.Join(", ", new List<string>{
                frontWallSpecularCB.Checked ? "Передняя" : null,
                backWallSpecularCB.Checked ? "Задняя" : null,
                leftWallSpecularCB.Checked ? "Левая" : null,
                rightWallSpecularCB.Checked ? "Правая" : null,
                upWallSpecularCB.Checked ? "Верхняя" : null,
                downWallSpecularCB.Checked ? "Нижняя" : null,
            }.Where(s => s != null));
            var sb = new StringBuilder();
            sb.AppendFormat("Фигур: {0} (Кубы: {1}, Шары: {2}) | Источники: {3}", figuresCount, cubesCount, spheresCount, lightsCount);
            if (!string.IsNullOrEmpty(specWalls)) sb.AppendFormat(" | Зеркальные стены: {0}", specWalls);
            if (cubeSpecularCB.Checked) sb.Append(" | Куб: прозрачный");
            if (sphereSpecularCB.Checked) sb.Append(" | Шар: прозрачный");
            if (refractCubeCB.Checked) sb.Append(" | Куб: зеркальный");
            if (refractSphereCB.Checked) sb.Append(" | Шар: зеркальный");
            statusLabel.Text = sb.ToString();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            // Инициализация формы (не используется)
        }

        private void groupBox2_Enter(object sender, EventArgs e)
        {
            // Обработчик UI (не используется)
        }

        private void groupBox3_Enter(object sender, EventArgs e)
        {
            // Обработчик UI (не используется)
        }

        private void groupBox1_Enter(object sender, EventArgs e)
        {
            // Обработчик UI (не используется)
        }

        private void twoLightsCB_CheckedChanged(object sender, EventArgs e)
        {
            // Переключение второго источника света (пересоздаёт сцену при рендере)
        }

        private void groupBox4_Enter(object sender, EventArgs e)
        {
            // Обработчик UI (не используется)
        }

        private void statusLabel_Click(object sender, EventArgs e)
        {
            // Клик по статусу (не используется)
        }

        // Главный цикл трассировки: генерирует пиксели, пускает лучи и записывает цвета
        public void run_rayTrace()
        {
            get_pixels();
            for(int i = 0; i < w; ++i)
                 for(int j = 0; j < h; ++j)
                 {
                    // Луч из камеры в точку экранной плоскости
                    Ray r = new Ray(focus, pixels[i, j]);
                    r.start = new Point3D(pixels[i, j]); // старт луча на плоскости (для корректной t)
                    Point3D clr = RayTrace(r, 10, 1);    // рекурсивная трассировка (глубина 10)
                    // Нормализация цвета, если компоненты > 1
                    if (clr.x > 1.0f || clr.y > 1.0f || clr.z > 1.0f)
                        clr = Point3D.norm(clr);
                    color_pixels[i, j] = Color.FromArgb((int)(255 * clr.x), (int)(255 * clr.y), (int)(255 * clr.z));
                 }
        }

        // Получение всех пикселей экранной плоскости по четырём углам
        public void get_pixels()
        {
            pixels = new Point3D[w, h];
            color_pixels = new Color[w, h];
            // Шаги вдоль верхнего и нижнего края для движения по X
            Point3D step_up = (up_right - up_left) / (w - 1);
            Point3D step_down = (down_right - down_left) / (w - 1);

            Point3D up = new Point3D(up_left);
            Point3D down = new Point3D(down_left);

            for (int i = 0; i < w; ++i)
            {
                // Для текущего столбца считаем шаг по Y (между верхней и нижней точками)
                Point3D step_y = (up - down) / (h - 1);
                Point3D d = new Point3D(down);
                for (int j = 0; j < h; ++j)
                {
                    pixels[i, j] = d;
                    d += step_y; // двигаемся вниз по колонке
                }
                // двигаемся вправо по верхнему и нижнему краю
                up += step_up;
                down += step_down;
            }
        }

        // Проверка, видима ли точка пересечения из источника света (нет ли преград)
        public bool is_visible(Point3D light_point, Point3D hit_point)
        {
            float max_t = (light_point - hit_point).length();     // позиция источника света на луче
            Ray r = new Ray(hit_point, light_point);               // теневой луч от точки к источнику

            foreach(Figure fig in scene)
                if (fig.figure_intersection(r, out float t, out Point3D n))
                    if (t < max_t && t > Figure.EPS)              // если пересечение до источника — есть тень
                        return false;
             return true;
        }

        // Рекурсивная трассировка луча: ближайшее пересечение + свет (ambient/diffuse) + отражение/преломление
        public Point3D RayTrace(Ray r, int iter, float env)
        {
            if (iter <= 0)
                return new Point3D(0, 0, 0);

            float t = 0;        // позиция точки пересечения луча с фигурой на луче
            Point3D normal = null; // нормаль в точке пересечения
            Material m = new Material();
            Point3D res_color = new Point3D(0, 0, 0);
            bool refract_out_of_figure = false; // луч преломления выходит из объекта?

            // Находим ближайшую фигуру по t
            foreach(Figure fig in scene)
            {
                if (fig.figure_intersection(r, out float intersect, out Point3D n))
                    if(intersect < t || t == 0)     // нужна ближайшая фигура к точке наблюдения
                    {
                        t = intersect;
                        normal = n;
                        m = new Material(fig.figure_material);
                    }
            }

            if (t == 0)
                return new Point3D(0, 0, 0);
            // Если угол между нормалью и направлением луча положительный — луч выходит из объекта (инвертируем нормаль)
            if (Point3D.scalar(r.direction, normal) > 0) 
            {
                normal *= -1; 
                refract_out_of_figure = true;
            }

            Point3D hit_point = r.start + r.direction * t; // точка попадания

            // Локальное освещение: ambient + diffuse (если точка видима источнику)
            foreach(Light l in lights)
            {
                Point3D amb = l.color_light * m.ambient;
                amb.x = (amb.x * m.clr.x);
                amb.y = (amb.y * m.clr.y);
                amb.z = (amb.z * m.clr.z);
                res_color += amb;

                // диффузное освещение
                if (is_visible(l.point_light, hit_point))
                    res_color += l.shade(hit_point, normal, m.clr, m.diffuse);
            }

            // Отражение
            if(m.reflection > 0)
            {
                Ray reflected_ray = r.reflect(hit_point, normal);
                res_color += m.reflection * RayTrace(reflected_ray, iter - 1, env);
            }

            // Преломление
            if(m.refraction > 0)
            {
                float eta;                 // коэффициент преломления
                if (refract_out_of_figure) // луч выходит в среду
                    eta = m.environment;   
                else
                    eta = 1 / m.environment; // вход в объект

                Ray refracted_ray = r.refract(hit_point, normal, eta);
                if(refracted_ray != null)
                    res_color += m.refraction * RayTrace(refracted_ray, iter - 1, m.environment);
            }

            return res_color;
        }
    }
}

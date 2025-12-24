using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CornellBoxRayTracerFinal
{
    // === ГЛАВНЫЙ КЛАСС ФОРМЫ ===
    public class MainForm : Form
    {
        private PictureBox pictureBox;
        private Bitmap renderBitmap;
        private Scene scene;

        // UI Элементы
        private CheckBox cbReflections;
        private CheckBox cbTransparency;
        private ComboBox comboMirrorWall;

        // Параметры
        private int width = 600;
        private int height = 600;

        public MainForm()
        {
            this.Text = "Cornell Box Ray Tracer (No Noise)";
            this.Size = new Size(880, 660);
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;

            // Холст для рисования
            pictureBox = new PictureBox();
            pictureBox.Location = new Point(10, 10);
            pictureBox.Size = new Size(width, height);
            pictureBox.BorderStyle = BorderStyle.Fixed3D;
            this.Controls.Add(pictureBox);

            // Инициализация сцены
            scene = new Scene();
            scene.InitScene();

            // === UI ПАНЕЛЬ УПРАВЛЕНИЯ ===
            int uiX = 630;
            int y = 20;

            // Кнопка Рендер
            Button btnRender = new Button() { Text = "Рендеринг", Location = new Point(uiX, y), Width = 200, Height = 40, BackColor = Color.LightBlue };
            btnRender.Click += (s, e) => Render();
            this.Controls.Add(btnRender);
            y += 55;

            // Группа: Эффекты
            GroupBox gbEffects = new GroupBox() { Text = "Настройки", Location = new Point(uiX, y), Width = 200, Height = 80 };
            cbReflections = new CheckBox() { Text = "Зеркальность", Location = new Point(10, 20), Checked = true, Width = 180 };
            cbTransparency = new CheckBox() { Text = "Прозрачность", Location = new Point(10, 45), Checked = true, Width = 180 };

            cbReflections.CheckedChanged += (s, e) => { scene.EnableReflections = cbReflections.Checked; Render(); };
            cbTransparency.CheckedChanged += (s, e) => { scene.EnableTransparency = cbTransparency.Checked; Render(); };

            gbEffects.Controls.Add(cbReflections);
            gbEffects.Controls.Add(cbTransparency);
            this.Controls.Add(gbEffects);
            y += 90;

            // Группа: Стены
            GroupBox gbWalls = new GroupBox() { Text = "Зеркальная стена", Location = new Point(uiX, y), Width = 200, Height = 60 };
            comboMirrorWall = new ComboBox() { Location = new Point(10, 25), Width = 180, DropDownStyle = ComboBoxStyle.DropDownList };
            comboMirrorWall.Items.AddRange(new string[] { "Нет (Матовая)", "Задняя", "Левая", "Правая", "Пол", "Потолок", "Передняя (за камерой)" });
            comboMirrorWall.SelectedIndex = 5; // Потолок по умолчанию
            comboMirrorWall.SelectedIndexChanged += (s, e) => { scene.MirrorWallIndex = comboMirrorWall.SelectedIndex; Render(); };
            gbWalls.Controls.Add(comboMirrorWall);
            this.Controls.Add(gbWalls);
            y += 70;

            // Группа: Доп. Свет
            GroupBox gbLight = new GroupBox() { Text = "Управление доп. светом", Location = new Point(uiX, y), Width = 200, Height = 100 };

            Button btnXPlus = new Button() { Text = "X+", Location = new Point(110, 20), Width = 40 };
            Button btnXMinus = new Button() { Text = "X-", Location = new Point(50, 20), Width = 40 };
            Label lblX = new Label() { Text = "X:", Location = new Point(10, 25), Width = 30 };

            Button btnYPlus = new Button() { Text = "Y+", Location = new Point(110, 45), Width = 40 };
            Button btnYMinus = new Button() { Text = "Y-", Location = new Point(50, 45), Width = 40 };
            Label lblY = new Label() { Text = "Y:", Location = new Point(10, 50), Width = 30 };

            Button btnZPlus = new Button() { Text = "Z+", Location = new Point(110, 70), Width = 40 };
            Button btnZMinus = new Button() { Text = "Z-", Location = new Point(50, 70), Width = 40 };
            Label lblZ = new Label() { Text = "Z:", Location = new Point(10, 75), Width = 30 };

            btnXPlus.Click += (s, e) => MoveLight(new Vec3(1, 0, 0));
            btnXMinus.Click += (s, e) => MoveLight(new Vec3(-1, 0, 0));
            btnYPlus.Click += (s, e) => MoveLight(new Vec3(0, 1, 0));
            btnYMinus.Click += (s, e) => MoveLight(new Vec3(0, -1, 0));
            btnZPlus.Click += (s, e) => MoveLight(new Vec3(0, 0, 1));
            btnZMinus.Click += (s, e) => MoveLight(new Vec3(0, 0, -1));

            gbLight.Controls.AddRange(new Control[] { btnXPlus, btnXMinus, lblX, btnYPlus, btnYMinus, lblY, btnZPlus, btnZMinus, lblZ });
            this.Controls.Add(gbLight);

            // Автозапуск рендера
            this.Shown += (s, e) => Render();
        }

        private void MoveLight(Vec3 dir)
        {
            if (scene.Lights.Count > 1)
            {
                scene.Lights[1].Position = scene.Lights[1].Position + dir * 1.5;
                Render();
            }
        }

        private void Render()
        {
            this.Text = "Rendering... (Подождите)";
            this.Cursor = Cursors.WaitCursor;

            renderBitmap = new Bitmap(width, height);

            BitmapData bData = renderBitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            int stride = bData.Stride;
            System.IntPtr scan0 = bData.Scan0;

            unsafe
            {
                byte* ptr = (byte*)scan0;

                Parallel.For(0, height, y =>
                {
                    for (int x = 0; x < width; x++)
                    {
                        // Координаты экрана
                        double u = (double)x / width * 2.0 - 1.0;
                        double v = -((double)y / height * 2.0 - 1.0);

                        // Камера
                        Vec3 origin = new Vec3(0, 5, -14);
                        Vec3 target = new Vec3(u * 5.5, v * 5.5 + 5, -9);
                        Vec3 direction = (target - origin).Normalize();

                        Ray ray = new Ray(origin, direction);

                        Vec3 color = scene.Trace(ray, 0);
                        color = Vec3.Clamp(color);

                        int offset = y * stride + x * 3;
                        ptr[offset + 0] = (byte)(color.z * 255); // B
                        ptr[offset + 1] = (byte)(color.y * 255); // G
                        ptr[offset + 2] = (byte)(color.x * 255); // R
                    }
                });
            }

            renderBitmap.UnlockBits(bData);
            pictureBox.Image = renderBitmap;
            this.Text = "Cornell Box Ray Tracer (Done)";
            this.Cursor = Cursors.Default;
        }

        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.Run(new MainForm());
        }
    }

    // === МАТЕМАТИКА ===
    public struct Vec3
    {
        public double x, y, z;
        public Vec3(double x, double y, double z) { this.x = x; this.y = y; this.z = z; }
        public static Vec3 operator +(Vec3 a, Vec3 b) => new Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
        public static Vec3 operator -(Vec3 a, Vec3 b) => new Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
        public static Vec3 operator *(Vec3 a, double d) => new Vec3(a.x * d, a.y * d, a.z * d);
        public static Vec3 operator *(Vec3 a, Vec3 b) => new Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
        public double Dot(Vec3 b) => x * b.x + y * b.y + z * b.z;
        public Vec3 Normalize() { double m = Math.Sqrt(x * x + y * y + z * z); return m == 0 ? new Vec3(0, 0, 0) : new Vec3(x / m, y / m, z / m); }
        public static Vec3 Clamp(Vec3 v) => new Vec3(Math.Min(1, Math.Max(0, v.x)), Math.Min(1, Math.Max(0, v.y)), Math.Min(1, Math.Max(0, v.z)));
    }

    public struct Ray
    {
        public Vec3 Origin;
        public Vec3 Direction;
        public Ray(Vec3 o, Vec3 d) { Origin = o; Direction = d; }
    }

    public class Light
    {
        public Vec3 Position;
        public Vec3 Color;
        public Light(Vec3 pos, Vec3 col) { Position = pos; Color = col; }
    }

    // === ФИГУРЫ ===
    public abstract class Shape
    {
        public Vec3 Color;
        public double Specular;
        public double Reflectivity;
        public double Transparency;

        public abstract double Intersect(Ray ray);
        public abstract Vec3 GetNormal(Vec3 point);
    }

    public class Sphere : Shape
    {
        public Vec3 Center;
        public double Radius;

        public Sphere(Vec3 c, double r, Vec3 col, double refl = 0, double transp = 0)
        {
            Center = c; Radius = r; Color = col;
            Reflectivity = refl; Transparency = transp; Specular = 0.6;
        }

        public override double Intersect(Ray ray)
        {
            Vec3 oc = ray.Origin - Center;
            double a = ray.Direction.Dot(ray.Direction);
            double b = 2.0 * oc.Dot(ray.Direction);
            double c = oc.Dot(oc) - Radius * Radius;
            double disc = b * b - 4 * a * c;

            if (disc < 0) return 0;
            double dist = (-b - Math.Sqrt(disc)) / (2.0 * a);
            if (dist > 0.001) return dist;
            return 0;
        }
        public override Vec3 GetNormal(Vec3 point) => (point - Center).Normalize();
    }

    public class Box : Shape
    {
        public Vec3 Min;
        public Vec3 Max;

        public Box(Vec3 min, Vec3 max, Vec3 col, double refl = 0, double transp = 0)
        {
            Min = min; Max = max; Color = col;
            Reflectivity = refl; Transparency = transp; Specular = 0.3;
        }

        public override double Intersect(Ray ray)
        {
            double t1 = (Min.x - ray.Origin.x) / ray.Direction.x;
            double t2 = (Max.x - ray.Origin.x) / ray.Direction.x;
            double t3 = (Min.y - ray.Origin.y) / ray.Direction.y;
            double t4 = (Max.y - ray.Origin.y) / ray.Direction.y;
            double t5 = (Min.z - ray.Origin.z) / ray.Direction.z;
            double t6 = (Max.z - ray.Origin.z) / ray.Direction.z;

            double tmin = Math.Max(Math.Max(Math.Min(t1, t2), Math.Min(t3, t4)), Math.Min(t5, t6));
            double tmax = Math.Min(Math.Min(Math.Max(t1, t2), Math.Max(t3, t4)), Math.Max(t5, t6));

            if (tmax < 0) return 0;
            if (tmin > tmax) return 0;
            return tmin > 0.001 ? tmin : 0;
        }

        public override Vec3 GetNormal(Vec3 p)
        {
            double eps = 0.001;
            if (Math.Abs(p.x - Min.x) < eps) return new Vec3(-1, 0, 0);
            if (Math.Abs(p.x - Max.x) < eps) return new Vec3(1, 0, 0);
            if (Math.Abs(p.y - Min.y) < eps) return new Vec3(0, -1, 0);
            if (Math.Abs(p.y - Max.y) < eps) return new Vec3(0, 1, 0);
            if (Math.Abs(p.z - Min.z) < eps) return new Vec3(0, 0, -1);
            if (Math.Abs(p.z - Max.z) < eps) return new Vec3(0, 0, 1);
            return new Vec3(0, 1, 0);
        }
    }

    public class Plane : Shape
    {
        public Vec3 Normal;
        public double Distance;

        public Plane(Vec3 n, double d, Vec3 col)
        {
            Normal = n.Normalize(); Distance = d; Color = col;
            Reflectivity = 0; Transparency = 0; Specular = 0;
        }

        public override double Intersect(Ray ray)
        {
            double denom = Normal.Dot(ray.Direction);
            if (Math.Abs(denom) > 0.0001)
            {
                double t = -(Normal.Dot(ray.Origin) + Distance) / denom;
                if (t > 0.001) return t;
            }
            return 0;
        }
        public override Vec3 GetNormal(Vec3 p) => Normal;
    }

    // === СЦЕНА ===
    public class Scene
    {
        public List<Shape> Objects = new List<Shape>();
        public List<Light> Lights = new List<Light>();

        public bool EnableReflections = true;
        public bool EnableTransparency = true;
        public int MirrorWallIndex = 5;

        Plane wallBack, wallLeft, wallRight, wallFloor, wallCeil, wallFront;

        public void InitScene()
        {
            // === СТЕНЫ ===
            wallBack = new Plane(new Vec3(0, 0, -1), 10, new Vec3(0.9, 0.9, 0.9));
            wallLeft = new Plane(new Vec3(1, 0, 0), 5, new Vec3(0.8, 0.1, 0.1));
            wallRight = new Plane(new Vec3(-1, 0, 0), 5, new Vec3(0.1, 0.8, 0.1));
            wallFloor = new Plane(new Vec3(0, 1, 0), 0, new Vec3(0.9, 0.9, 0.9));
            wallCeil = new Plane(new Vec3(0, -1, 0), 10, new Vec3(0.9, 0.9, 0.9));
            wallFront = new Plane(new Vec3(0, 0, 1), 16, new Vec3(0.9, 0.9, 0.9));

            Objects.AddRange(new Shape[] { wallBack, wallLeft, wallRight, wallFloor, wallCeil, wallFront });

            // === ОБЪЕКТЫ ===

            // 1. Сфера (Зеркальная)
            Objects.Add(new Sphere(new Vec3(-2.5, 2.5, 7), 2.5, new Vec3(0.1, 0.1, 0.1), refl: 0.85, transp: 0));

            // 2. Сфера 2 (Матовая Синяя)
            Objects.Add(new Sphere(new Vec3(2.5, 1.5, 6), 1.5, new Vec3(0.2, 0.2, 0.8), refl: 0, transp: 0));

            // 3. Куб 1 (Прозрачный)
            Objects.Add(new Box(new Vec3(2, 0, 1), new Vec3(4, 2, 3), new Vec3(0.9, 0.9, 0.9), refl: 0.1, transp: 0.9));

            // 4. Куб 2 (Желтый Матовый)
            Objects.Add(new Box(new Vec3(-4, 0, 2), new Vec3(-2, 2, 4), new Vec3(0.9, 0.8, 0.2), refl: 0.0, transp: 0.0));

            // === СВЕТ ===
            Lights.Add(new Light(new Vec3(0, 9.8, 5), new Vec3(1.1, 1.1, 1.1)));
            Lights.Add(new Light(new Vec3(0, 5, 0), new Vec3(0.5, 0.4, 0.3)));
        }

        public Vec3 Trace(Ray ray, int depth)
        {
            if (depth > 5) return new Vec3(0, 0, 0);

            UpdateWallMaterials();

            double tMin = double.MaxValue;
            Shape hitObj = null;

            foreach (var obj in Objects)
            {
                double t = obj.Intersect(ray);
                if (t > 0 && t < tMin)
                {
                    tMin = t;
                    hitObj = obj;
                }
            }

            if (hitObj == null) return new Vec3(0, 0, 0);

            Vec3 hitPoint = ray.Origin + ray.Direction * tMin;
            Vec3 normal = hitObj.GetNormal(hitPoint);

            Vec3 bias = normal * 0.02;

            Vec3 finalColor = new Vec3(0, 0, 0);
            finalColor = finalColor + hitObj.Color * 0.15; // Ambient

            foreach (var light in Lights)
            {
                Vec3 lightDir = (light.Position - hitPoint);
                double dist = Math.Sqrt(lightDir.Dot(lightDir));
                lightDir = lightDir.Normalize();

                bool inShadow = false;
                Ray shadowRay = new Ray(hitPoint + bias, lightDir);
                foreach (var obj in Objects)
                {
                    if (obj == hitObj) continue;
                    if (EnableTransparency && obj.Transparency > 0.5) continue;

                    double t = obj.Intersect(shadowRay);
                   
                    if (t > 0.001 && t < dist)
                    {
                        inShadow = true;
                        break;
                    }
                }

                if (!inShadow)
                {
                    double diff = Math.Max(0, normal.Dot(lightDir));
                    finalColor = finalColor + hitObj.Color * light.Color * diff;

                    Vec3 viewDir = (ray.Origin - hitPoint).Normalize();
                    Vec3 halfDir = (lightDir + viewDir).Normalize();
                    double spec = Math.Pow(Math.Max(0, normal.Dot(halfDir)), 50);
                    if (hitObj.Specular > 0)
                        finalColor = finalColor + light.Color * spec * hitObj.Specular;
                }
            }

            // Отражение
            if (EnableReflections && hitObj.Reflectivity > 0)
            {
                Vec3 rDir = ray.Direction - normal * 2.0 * normal.Dot(ray.Direction);
                Ray reflectRay = new Ray(hitPoint + bias, rDir);
                Vec3 rCol = Trace(reflectRay, depth + 1);

                finalColor = finalColor * (1 - hitObj.Reflectivity) + rCol * hitObj.Reflectivity;
            }

            // Прозрачность
            if (EnableTransparency && hitObj.Transparency > 0)
            {
                Ray transRay = new Ray(hitPoint - normal * 0.002, ray.Direction);
                Vec3 tCol = Trace(transRay, depth + 1);
                finalColor = finalColor * (1 - hitObj.Transparency) + tCol * hitObj.Transparency;
            }

            return finalColor;
        }

        private void UpdateWallMaterials()
        {
            wallBack.Reflectivity = 0; wallLeft.Reflectivity = 0;
            wallRight.Reflectivity = 0; wallFloor.Reflectivity = 0;
            wallCeil.Reflectivity = 0; wallFront.Reflectivity = 0;

            if (!EnableReflections) return;

            switch (MirrorWallIndex)
            {
                case 1: wallBack.Reflectivity = 0.8; break;
                case 2: wallLeft.Reflectivity = 0.8; break;
                case 3: wallRight.Reflectivity = 0.8; break;
                case 4: wallFloor.Reflectivity = 0.8; break;
                case 5: wallCeil.Reflectivity = 0.8; break;
                case 6: wallFront.Reflectivity = 0.8; break;
            }
        }
    }
}
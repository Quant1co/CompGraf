using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RayTracing
{
    // Параметры материала поверхности для модели освещения и рейтрейсинга
    public class Material
    {
        public float reflection;    // коэффициент отражения (вклад отражённого луча)
        public float refraction;    // коэффициент преломления (вклад преломлённого луча)
        public float environment;   // показатель преломления среды (n)
        public float ambient;       // коэффициент фонового освещения (ambient)
        public float diffuse;       // коэффициент диффузного освещения (Lambert)
        public Point3D clr;         // цвет материала (RGB в [0,1])

        public Material(float refl, float refr, float amb, float dif, float env = 1)
        {
            reflection = refl; 
            refraction = refr;
            ambient = amb;
            diffuse = dif; 
            environment = env;
        }

        // Копирующий конструктор
        public Material(Material m)
        {
            reflection = m.reflection;
            refraction = m.refraction;
            environment = m.environment;
            ambient = m.ambient;
            diffuse = m.diffuse;
            clr = new Point3D(m.clr);
        }

        public Material() { }
    }
}

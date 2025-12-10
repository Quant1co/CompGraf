using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RayTracing
{
    // Луч: стартовая точка и нормализованное направление, с операциями отражения и преломления
    public class Ray
    {
        public Point3D start, direction; // начало и направление луча

        public Ray(Point3D st, Point3D end)
        {
            start = new Point3D(st);
            direction = Point3D.norm(end - st); // нормализуем направление от start к end
        }

        public Ray() { } 

        public Ray(Ray r)
        {
            start = r.start;
            direction = r.direction;
        }

        // Отражение: R = D - 2N(D•N)
        public Ray reflect(Point3D hit_point, Point3D normal)
        {
            Point3D reflect_dir = direction - 2 * normal * Point3D.scalar(direction, normal);
            return new Ray(hit_point, hit_point + reflect_dir);
        }

        // Преломление по формуле Снеллиуса в векторной форме
        // eta — отношение показателей преломления (вход/выход)
        public Ray refract(Point3D hit_point, Point3D normal, float eta)
        {
            Ray res_ray = new Ray();
            float sclr = Point3D.scalar(normal, direction); // косинус угла между нормалью и направлением

            float k = 1 - eta * eta * (1 - sclr * sclr);   // проверка на полное внутреннее отражение

            if (k >= 0)
            {
                float cos_theta = (float)Math.Sqrt(k);
                res_ray.start = new Point3D(hit_point);
                // направление преломлённого луча
                res_ray.direction = Point3D.norm(eta * direction - (cos_theta + eta * sclr) * normal);
                return res_ray;
            }
            else
                return null; // полное внутреннее отражение — преломлённого луча нет
        }
    }
}

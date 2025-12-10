using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace RayTracing
{
    static class Program
    {
        /// <summary>
        /// Главная точка входа для приложения WinForms.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());  // запуск главной формы с рейтрейсером
        }
    }
}

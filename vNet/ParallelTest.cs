using System;
using System.Threading.Tasks;

namespace vNet
{
    internal class ParallelTest
    {
        public static void PTest()
        {
            var timer = new System.Diagnostics.Stopwatch();

            timer.Restart();

            int n = 10000, l = 100000;

            void kernel(int i)
            {
                Console.WriteLine(i);
            }

            Parallel.For(0, n, kernel);
            timer.Stop();
            Console.WriteLine(timer.ElapsedMilliseconds);
            Console.ReadKey();
            timer.Restart();
            for (int i = 0; i < n; i++)
            {
                Console.WriteLine(i * i + i ^ 2);
            }
            timer.Stop();
            Console.WriteLine(timer.ElapsedMilliseconds);
            Console.ReadKey();
        }
    }
}
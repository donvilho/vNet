using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using vNet.Dll;

namespace vNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var a = Utils.GenerateMatrix(28, 28, 1, true);

            var b = Utils.InitKernels(3, 3, 10);

            //Utils.DrawFromArray(a);

            var c = Operations.ApplyPadding(a, 2);

            //Utils.DrawFromArray(c);

            var timer = new Stopwatch();

            timer.Restart();
            var d = Operations.Convolution2D(a, b, stride: 1, padding: 1);

            var mpool = Operations.MaxPool(d, 2, 2);

            dynamic eka;

            for (int i = 0; i < 100; i++)
            {
                //Console.WriteLine(i);
                eka = Operations.Convolution2D(a, b, stride: 1, padding: 1);
            }

            timer.Stop();

            Console.WriteLine("ms: " + timer.ElapsedMilliseconds);

            timer.Restart();

            timer.Stop();

            Console.WriteLine("ms: " + timer.ElapsedMilliseconds);

            //int result = cpp.Addf(val1, val2);

            //Console.WriteLine(result);

            var Model = new LogisticRegression();

            Model.TrainModel(
               path: "MnistDouble.bin",
                   epoch: 40,
                   learningRate: 0.1d,
                   stepDecay: 0,
                   momentum: 0.0f,
                   miniBatch: 256,
                   l2: false);

            //Model.RunModel(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\c");

            Console.ReadKey();
            Console.ReadKey();
        }
    }
}
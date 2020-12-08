using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

namespace vNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var Model = new LogisticRegression();

            Model.TrainModel(
               path: "MnistDouble.bin",
               epoch: 40,
               learningRate: 0.1d,
               stepDecay: 0,
               momentum: 0.0f,
               miniBatch: 0,
               l2: false);

            //Model.RunModel(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\c");

            Console.ReadKey();
            Console.ReadKey();
        }
    }
}
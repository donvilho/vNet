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

            //Model.MultiTraining("MnistFull.bin");

            Model.TrainModel(
               path: "xray.bin",
               epoch: 400,
               learningRate: 0.001f,
               stepDecay: 0,
               momentum: 0.0f,
               miniBatch: 10,
               l2: true);

            //Model.RunModel(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\c");

            Console.ReadKey();
        }
    }
}
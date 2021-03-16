using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace vNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var parentPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName;
            var dataset = new Dataset(parentPath + @"/Data/training", parentPath + @"/Data/testing");
            var Model = new LogisticRegression(dataset);

            Model.TrainModel(
                   epoch: 40,
                   learningRate: 0.1d,
                   stepDecay: 0,
                   momentum: 0.0f,
                   miniBatch: 256,
                   l2: false);

            //Model.RunModel(@"");

            Console.ReadKey();
        }
    }
}
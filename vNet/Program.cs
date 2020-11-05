using ICSharpCode.SharpZipLib;
using ScottPlot;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace vNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var trainingset = Utils.DataArrayCreator(@"C:\Users\Viert\Downloads\mnist_png.tar\mnist_png\training\");
            var testset = Utils.DataArrayCreator(@"C:\Users\Viert\Downloads\mnist_png.tar\mnist_png\testing\");
            var Dataset = new Dataset(trainingset, testset);

            Dataset.Reduce(20);

            var logReg = new LogisticRegression(5000000, .01f);
            logReg.TrainModel(Dataset);
        }
    }
}
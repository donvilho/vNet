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
    class Program
    {
         static void Main(string[] args)
       {

            Console.WriteLine(Math.Exp(0));

            var logReg = new LogisticRegression(600, .001f);
            logReg.TrainModel(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png",20);

        }
    }
}

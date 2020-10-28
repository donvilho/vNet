using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data;
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

            //ParallelTest.PTest();

            //var linearDataset = new Dataset(Utils.CSVtoArray(@"C:\Users\ville\Downloads\lohi.csv").ToArray());

            //var test = new DatasetArray(@"C:\Users\ville\Downloads\mnist_png.tar\linear");

            //var dataset = Utils.DatasetCreator(@"C:\Users\ville\Downloads\mnist_png.tar\linear");

            var logReg = new LogisticRegression(200, .01f, 32);

            

            logReg.TrainModel(@"C:\Users\ville\Downloads\mnist_png.tar\linear");

            //var linearReg = new LinearRegression(linearDataset, 10, 0.01f);

            //linearReg.TrainModel();


            

            //var structure = new List<(int, Activator)> { (1,Activator.None) };

            //var net = new NetworkTrainer(dataset.InputLenght, structure);
         

            //net.Train(dataset, learningRate: .001f, epoch:1000, costFunction: CostFunction.MSE, miniBatch: 0);

           

        }



        
    }
}

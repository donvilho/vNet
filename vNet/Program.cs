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


            // Source must be array or IList.
            var source = Enumerable.Range(0, 100000).ToArray();

            // Partition the entire source array.
            var rangePartitioner = Partitioner.Create(0, source.Length);

            double[] results = new double[source.Length];

            // Loop over the partitions in parallel.
            Parallel.ForEach(rangePartitioner, (range, loopState) =>
            {
                // Loop over each range element without a delegate invocation.
                for (int i = range.Item1; i < range.Item2; i++)
                {
                    results[i] = source[i] * Math.PI;
                }
            });

            Console.WriteLine("Operation complete. Print results? y/n");
            char input = Console.ReadKey().KeyChar;
            if (input == 'y' || input == 'Y')
            {
                foreach (double d in results)
                {
                    Console.Write("{0} ", d);
                }
            }




            //ParallelTest.PTest();

            //var linearDataset = new Dataset(Utils.CSVtoArray(@"C:\Users\ville\Downloads\lohi.csv").ToArray());

            //var test = new DatasetArray(@"C:\Users\ville\Downloads\mnist_png.tar\linear");

            //var dataset = Utils.DatasetCreator(@"C:\Users\ville\Downloads\mnist_png.tar\linear");

            var logReg = new LogisticRegression(200, .01f, 128);
            
            
            

            logReg.TrainModel(@"C:\Users\Viert\Downloads\mnist_png.tar\mnist_png");

            //var linearReg = new LinearRegression(linearDataset, 10, 0.01f);

            //linearReg.TrainModel();


            

            //var structure = new List<(int, Activator)> { (1,Activator.None) };

            //var net = new NetworkTrainer(dataset.InputLenght, structure);
         

            //net.Train(dataset, learningRate: .001f, epoch:1000, costFunction: CostFunction.MSE, miniBatch: 0);

           

        }



        
    }
}

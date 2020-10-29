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

            var thre = new ThreadLocal<TT>(() => new TT(),true);

            Parallel.For(0, 10, i =>
            {

                thre.Value.print(i);
            });


            var result = thre.Values.ToArray();
     
        
            Console.WriteLine(result); 

            //var linearDataset = new Dataset(Utils.CSVtoArray(@"C:\Users\ville\Downloads\lohi.csv").ToArray());

            //var test = new DatasetArray(@"C:\Users\ville\Downloads\mnist_png.tar\linear");

            //var dataset = Utils.DatasetCreator(@"C:\Users\ville\Downloads\mnist_png.tar\linear");

            var logReg = new LogisticRegression(3, .01f, 256);
            
            
            

            logReg.TrainModel(@"C:\Users\ville\Downloads\mnist_png.tar\linear");

            //var linearReg = new LinearRegression(linearDataset, 10, 0.01f);

            //linearReg.TrainModel();


            

            //var structure = new List<(int, Activator)> { (1,Activator.None) };

            //var net = new NetworkTrainer(dataset.InputLenght, structure);
         

            //net.Train(dataset, learningRate: .001f, epoch:1000, costFunction: CostFunction.MSE, miniBatch: 0);

           

        }



        
    }

    class TT
    {
        public int Num;

        public TT()
        {
            Num = 0;
        }

        public void print(int i)
        {
           
            Num += i;
            Console.WriteLine("th: " + Thread.CurrentThread.ManagedThreadId+" numA = "+(Num-i)+" num = "+Num);

        }
    }
}

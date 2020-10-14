using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
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
            var epoch = 100;
            var learningRate = 1;
            var datasetPath = @"C:\Users\ville\Downloads\mnist_png.tar\linear";

            var layers = new Layer[]
            {
                new Layer(10,Activation.Sigmoid),
                new Layer(200,Activation.None)
            };

            var model = new Model(Type.Lin_Reg, CostFunction.Msqrt, datasetPath, epoch, learningRate);


            model.TrainNetwork(epoch:10 , learningRate:.0001);


            Console.ReadKey();

        }


       

    }
}

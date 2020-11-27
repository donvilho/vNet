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
            //var trainingset = @"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\training";
            //var testset = @"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\testing";
            //var dataset = new Dataset(trainingset, testset);

            var dataset = Utils.DatasetFromBinary("MnistFull.bin");

            //dataset.ReduceToPercentage(10);

            var Model = new LogisticRegression(
                dataset,
                DropoutLowerThreshold: 0,
                DropoutUpperThreshold: 0);

            //Model.MultiTraining();

            Model.TrainModel(epoch: 300,
               learningRate: .01f,
               stepDecay: 150,
               momentum: .1f,
               miniBatch: 128,
               l2: true);

            Model.RunModel(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\c");

            Console.ReadKey();

            //Model.MultiTraining();

            /*
            Model.TrainModel(epoch: 20,
                learningRate: .01f,
                stepDecay: 50,
                momentum: .0f,
                miniBatch: 32);
            */
            // Hyvä esimerkki datasta
            /*
            Dataset.ReduceToPercentage(20);

            var Model = new LogisticRegression(Dataset,
                DropoutLowerThreshold: 1,
                DropoutUpperThreshold: 0,
                constInit: false);

            Model.TrainModel(epoch: 150,
                learningRate: .01f,
                stepDecay: 50,
                momentum: .0f,
                miniBatch: 128);
            */
            //var Model = new LogisticRegression(Dataset, DropoutThreshold: 2000, constInit: true);
            //Model.TrainModel(epoch: 100, learningRate: .125f, momentum: 0.75f, miniBatch: 128, validatewithTS: false);

            //Hyväveto tämäkin
            //var Model = new LogisticRegression(Dataset, DropoutThreshold: 2000, constInit: true);
            //Model.TrainModel(epoch: 190, learningRate: .1f, momentum: 0.5f, miniBatch: 128, validatewithTS: false);

            // 90's
            //Dataset.ReduceToPercentage(50);
            //var Model = new LogisticRegression(Dataset, DropoutThreshold: 2000, constInit: true);
            //Model.TrainModel(epoch: 50, learningRate: .1f, momentum: 0.5f, miniBatch: 128, validatewithTS: false);

            // 91%
            //Dataset.ReduceToPercentage(30);
            //var Model = new LogisticRegression(Dataset, DropoutThreshold: 1, constInit: true);
            //Model.TrainModel(epoch: 50, learningRate: .1f, momentum: 0.5f, miniBatch: 128, validatewithTS: false);

            //Model.TrainModel(epoch: 100, learningRate: .1f, momentum: 0.5f, miniBatch: 128, validatewithTS: false);
            //var Model = new LogisticRegression(Dataset, DropoutThreshold: 200, constInit: true);
            //Model.TrainModel(epoch: 300, learningRate: .1f, momentum: 0.5f, miniBatch: 256, validatewithTS: false);
            //Model.TrainModel(epoch: 300, learningRate: 0.01f, momentum: 1f, miniBatch: 256); //OK veto
        }
    }
}
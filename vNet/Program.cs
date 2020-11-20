using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Threading.Tasks;

namespace vNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var trainingset = Utils.DataArrayCreator(@"C:\Users\Viert\Downloads\mnist_png.tar\mnist_png\training");
            var testset = Utils.DataArrayCreator(@"C:\Users\Viert\Downloads\mnist_png.tar\mnist_png\testing");
            var Dataset = new Dataset(trainingset, testset);

            //Dataset.Reduce(30);

            var Model = new LogisticRegression(Dataset, DropoutThreshold: 2000, constInit: true);
            Model.TrainModel(epoch: 100, learningRate: .1f, momentum: 0.5f, miniBatch: 256, validatewithTS: false);

            //var Model = new LogisticRegression(Dataset, DropoutThreshold: 200, constInit: true);
            //Model.TrainModel(epoch: 300, learningRate: .1f, momentum: 0.5f, miniBatch: 256, validatewithTS: false);
            //Model.TrainModel(epoch: 300, learningRate: 0.01f, momentum: 1f, miniBatch: 256); //OK veto
        }
    }
}
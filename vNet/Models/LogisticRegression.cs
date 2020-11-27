using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using vNet.Activations;
using vNet.LossFunctions;

namespace vNet
{
    public class LogisticRegression
    {
        public Dataset Data { get; set; }

        public Trainer NetModel;

        public LogisticRegression(Dataset dataset, int DropoutLowerThreshold = 0, int DropoutUpperThreshold = 0)
        {
            Data = dataset;
            NetModel = null;
            // Connection mask

            if (DropoutLowerThreshold != 0 | DropoutUpperThreshold != 0)
            {
                var temp = new List<int>();
                var interMidLayer = new float[dataset.InputLenght];

                for (int i = 0; i < dataset.TrainingData.Length; i++)
                {
                    for (int j = 0; j < dataset.InputLenght; j++)
                    {
                        interMidLayer[j] += dataset.TrainingData[i].Data[j] > 0 ? 1 : 0;
                    }
                }

                /*
                for (int i = 0; i < interMidLayer.Length; i++)
                {
                    interMidLayer[i] /= dataset.TrainingData.Length;
                }
                */

                for (int i = 0; i < interMidLayer.Length; i++)
                {
                    if (interMidLayer[i] > DropoutLowerThreshold | interMidLayer[i] < DropoutUpperThreshold)
                    {
                        temp.Add(i);
                    }
                }

                Data.ApplyConnectionMask(temp.ToArray());
            }
        }

        public void RunModel(string path)
        {
            Console.WriteLine("\n\nRunning model for file or files in path " + path + "\n");

            if (NetModel != null)
            {
                NetModel.Run(path);
            }

            Process.Start(path);
        }

        public void TrainModel(int epoch, float learningRate, bool l2, int stepDecay, float momentum, int miniBatch)
        {
            if (miniBatch == 0) { miniBatch = Data.TrainingData.Length; }

            Console.WriteLine("-----Starting training-----\n" +
                "<-Parameters->\n" +
                "Epoch: {0}\n" +
                "Batchsize: {2}\n" +
                "Learningrate: {1}\n" +
                "Momentum: {3}\n",
                epoch, learningRate, miniBatch, momentum);

            var trainer = new Trainer(Data);
            trainer.Init(learningRate, miniBatch, epoch, momentum, stepDecay, l2);
            //var result = trainer.Train(Data, epoch, learningRate, miniBatch, momentum, stepDecay, l2);

            trainer.Train(Data, true, false);

            var result = trainer.GetResult();

            Plot.Graph(result.Item2, learningRate, miniBatch, result.Item6);

            NetModel = trainer;
        }

        public void MultiTraining()
        {
            var l2 = new bool[] { true, false };
            var lrs = new float[] { .1f, 0.01f };
            var bts = new int[] { 8, 16, 32, 64, 128, 256, 512 };
            var epochs = new int[] { 50 };
            var moms = new float[] { 0.1f, .5f, 0f };
            var decay = new int[] { 50 };

            var Models = new List<Trainer>();

            for (int i = 0; i < lrs.Length; i++)
            {
                for (int j = 0; j < bts.Length; j++)
                {
                    for (int k = 0; k < epochs.Length; k++)
                    {
                        for (int l = 0; l < moms.Length; l++)
                        {
                            for (int m = 0; m < l2.Length; m++)
                            {
                                for (int n = 0; n < decay.Length; n++)
                                {
                                    //combinations.Add((lrs[i], bts[j], epochs[k], moms[l], l2[m]));
                                    //new Trainer(Data).Init(param.Item1, param.Item2, param.Item3, param.Item4, param.Item3, param.Item5);
                                    var mdl = new Trainer(Data);
                                    mdl.Init(lrs[i], bts[j], epochs[k], moms[l], decay[n], l2[m]);
                                    Models.Add(mdl);
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine("Total models running: " + Models.Count());

            GC.Collect();

            Parallel.ForEach(Models, (model) =>
            {
                model.Train(Data, false, true);
            });

            var Results = new List<(float, double[,], float, float, int, int, bool)>();

            Models.ToList().ForEach(x => Results.Add(x.GetResult()));

            var BestModel = Models.OrderByDescending(x => x.HighestResult).First();

            Console.WriteLine(BestModel.GetResult().ToString());

            Plot.GraphList(Results);

            NetModel = BestModel;

            //BestModel.FocusedTraining(Data);
        }
    }
}
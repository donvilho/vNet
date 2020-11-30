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
        public Trainer NetModel;

        public LogisticRegression()
        {
            NetModel = null;
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

        public void TrainModel(string path, int epoch, float learningRate, bool l2, int stepDecay, float momentum, int miniBatch)
        {
            Console.WriteLine("-----Starting training-----\n" +
                "<-Parameters->\n" +
                "Epoch: {0}\n" +
                "Batchsize: {2}\n" +
                "Learningrate: {1}\n" +
                "Momentum: {3}\n",
                epoch, learningRate, miniBatch, momentum);

            var trainer = new Trainer(path);
            trainer.Init(learningRate, miniBatch, epoch, momentum, stepDecay, l2);
            //var result = trainer.Train(Data, epoch, learningRate, miniBatch, momentum, stepDecay, l2);

            trainer.Train(true, false);

            var result = trainer.GetResult();

            Plot.Graph(result.Item2, learningRate, miniBatch, result.Item6);

            trainer.PlotModel();

            NetModel = trainer;
        }

        public void MultiTraining(string path)
        {
            var l2 = new bool[] { false };
            var lrs = new float[] { 0.01f, };
            var bts = new int[] { 16, 32, 64, 128 };
            var epochs = new int[] { 50 };
            var moms = new float[] { 0f };
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
                                    var mdl = new Trainer(path);
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
                model.Train(false, false);
            });

            var Results = new List<(float, double[,], float, float, int, int, bool)>();

            Models.ToList().ForEach(x => Results.Add(x.GetResult()));

            var BestModel = Models.OrderByDescending(x => x.HighestResult).First();

            Console.WriteLine("Highest model:\n" + BestModel.GetValue());

            Plot.GraphList(Results);

            NetModel = BestModel;

            //BestModel.FocusedTraining(Data);
        }
    }
}
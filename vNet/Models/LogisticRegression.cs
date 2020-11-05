using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;

namespace vNet
{
    internal class LogisticRegression : IModel
    {
        private int Epoch;
        private float LearningRate;
        private int MiniBatch;
        private double[,] PlotData;
        private List<(int, int)> Heatmap;
        private float Acc;
        private int Classes;
        private Network Net;

        public LogisticRegression(int epoch, float learningrate, int minibatch = 0)
        {
            Epoch = epoch;
            LearningRate = learningrate;
            MiniBatch = minibatch;
            PlotData = new double[Epoch, 2];
            Heatmap = new List<(int, int)>();
        }

        public void TestModel()
        {
            throw new NotImplementedException();
        }

        public unsafe void TrainModel(Dataset Dataset, bool plot = false)
        {
            var neuronCount = Dataset.OutputLenght;
            var inputLenght = Dataset.InputLenght;
            Classes = neuronCount;

            Net = new Network(neuronCount, inputLenght);

            if (MiniBatch == 0) { MiniBatch = Dataset.TrainingData.Length; }

            int BatchCount = 0;

            var trainer = new Trainer(Net);

            // Main Loop

            for (int e = 0; e < Epoch; e++)
            {
                Dataset.Shuffle(Dataset.TrainingData);

                foreach (var input in Dataset.TrainingData)
                {
                    trainer.Train(input);
                    BatchCount++;

                    if (BatchCount == MiniBatch)
                    {
                        Net.UpdateWeights(MiniBatch, LearningRate);
                        BatchCount = 0;
                    }
                }

                TestNet(Dataset.ValidationgData, e);
                e++;
            }

            TestNet(Dataset.ValidationgData, Epoch, true);

            Plot.Graph(PlotData, LearningRate, MiniBatch);
        }

        private void TestNet(Input[] Data, int epoch, bool plot = false)
        {
            var trainer = new Trainer(Net);
            var classcount = new int[Classes];

            if (plot)
            {
                var faults = new int[Classes];
                var correct = new int[Classes];
                var plt = new ScottPlot.Plot(600, 400);
                var defaultMarker = 10;
                Heatmap.Clear();

                foreach (var input in Data)
                {
                    var result = trainer.Test(input);

                    var yPos = 0;

                    for (int i = 0; i < input.TruthLabel.Length; i++)
                    {
                        if (input.TruthLabel[i] == 1)
                        {
                            yPos = i;
                            classcount[i]++;
                        }
                    }
                    Heatmap.Add((yPos, result.Item3));
                }

                foreach (var item in Heatmap)
                {
                    if (item.Item1 == item.Item2)
                    {
                        correct[item.Item1]++;
                        var marker = defaultMarker * (correct[item.Item1] * 0.007);
                        plt.PlotPoint(item.Item1, item.Item2, markerSize: marker, color: Color.Green);
                    }
                    else
                    {
                        faults[item.Item1]++;

                        var marker = defaultMarker * (faults[item.Item1] * 0.007);

                        plt.PlotPoint(item.Item1, item.Item2, markerSize: marker, color: Color.Black);
                    }
                }

                plt.Grid(xSpacing: 1, ySpacing: 1);
                plt.SaveFig("Experimental_Heatmap_HeatmapImage.png");
                Process.Start(new ProcessStartInfo("Experimental_Heatmap_HeatmapImage.png") { UseShellExecute = true });
            }
            else
            {
                var TestError = 0f;
                var Accuracy = 0f;

                foreach (var input in Data)
                {
                    var result = trainer.Test(input);

                    TestError += result.Item1;

                    Accuracy += result.Item2 == true ? 1 : 0;
                }

                Acc = Accuracy / Data.Length;

                PlotData[epoch, 0] = TestError / Data.Length;
                PlotData[epoch, 1] = Acc;
                Console.WriteLine("Epoch: " + epoch + " Acccuracy: " + Acc + " Error: " + TestError / Data.Length);
            }
        }
    }
}
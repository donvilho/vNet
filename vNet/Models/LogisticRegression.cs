using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using vNet.Activations;
using vNet.LossFunctions;

namespace vNet
{
    internal class LogisticRegression : ModelType
    {
        private int Epoch;
        private float LearningRate;
        private int MiniBatch;
        private double[,] PlotData;
        private List<(int, int)> Heatmap;
        private float Acc;
        private int Classes;
        private float[] Output;

        private Activation activation;
        private Loss loss;

        private float HighestResult;
        private int HighestResultEpoch;

        private Dataset Data;

        private Neuron[] Neurons;

        public LogisticRegression(Dataset dataset)
        {
            HighestResult = 0;
            HighestResultEpoch = 0;
            Output = new float[Classes];
            Data = dataset;
            Classes = dataset.classCount;
            Heatmap = new List<(int, int)>();
            Neurons = new Neuron[Classes];

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(Data.InputLenght);
            }
        }

        public void TrainModel(int epoch, float learningRate, int miniBatch = 0)
        {
            Epoch = epoch;
            LearningRate = learningRate;
            MiniBatch = miniBatch;
            PlotData = new double[Epoch, 2];

            if (Classes > 2)
            {
                activation = new Softmax();
                loss = new CrossEntropy();
            }
            else
            {
                activation = new Sigmoid();
                loss = new CrossEntropy();
            }

            Trainer();

            Plot.Graph(PlotData, LearningRate, MiniBatch, HighestResultEpoch);
        }

        private void TestModel(int epoch, bool plot = false)
        {
            var Prediction = new float[Classes];
            var classcount = new int[Classes];

            if (plot)
            {
                var faults = new int[Classes];
                var correct = new int[Classes];
                var plt = new ScottPlot.Plot(600, 400);
                var defaultMarker = 10;
                Heatmap.Clear();

                foreach (var input in Data.ValidationgData)
                {
                    //forward
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].ForwardCalculation(input.Data);
                    }

                    //activate
                    Output = activation.Activate(Neurons);

                    // Convert output
                    int position = Output.ToList().IndexOf(Output.Max());

                    var yPos = input.TruthLabel.ToList().IndexOf(input.TruthLabel.Max());
                    classcount[yPos]++;

                    Heatmap.Add((yPos, position));
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
                var Loss = 0f;
                var Accuracy = 0f;

                foreach (var input in Data.ValidationgData)
                {
                    //forward
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].ForwardCalculation(input.Data);
                    }

                    //activate
                    Output = activation.Activate(Neurons);

                    Loss += -loss.Calculate(Output, input.TruthLabel);

                    // Convert output
                    int position = Output.ToList().IndexOf(Output.Max());
                    var yPos = input.TruthLabel.ToList().IndexOf(input.TruthLabel.Max());

                    Accuracy += position == yPos ? 1 : 0;
                }

                Acc = (float)Accuracy / Data.ValidationgData.Length;

                if (Acc > HighestResult)
                {
                    HighestResult = Acc;
                    HighestResultEpoch = epoch;
                }

                PlotData[epoch, 0] = Loss / Data.ValidationgData.Length;
                PlotData[epoch, 1] = Acc;
                Console.WriteLine("Epoch: " + epoch + " Acccuracy: " + Acc + " Error: " + Loss / Data.ValidationgData.Length);
            }
        }

        private void Trainer()
        {
            var Output = new float[Classes];

            if (MiniBatch == 0) { MiniBatch = Data.TrainingData.Length; }

            int BatchCount = 0;

            // Main Loop

            for (int e = 0; e < Epoch; e++)
            {
                Data.Shuffle(Data.TrainingData);

                //Training loop
                foreach (var input in Data.TrainingData)
                {
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].ForwardCalculation(input.Data);
                    }

                    Output = activation.Activate(Neurons);

                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].Derivate = activation.Derivate(Output[i], input.TruthLabel[i]);
                        Neurons[i].Backpropagate(input.Data);
                    }

                    BatchCount++;

                    if (BatchCount == MiniBatch)
                    {
                        for (int i = 0; i < Neurons.Length; i++)
                        {
                            Neurons[i].AdjustWeights(MiniBatch, LearningRate);
                        }

                        BatchCount = 0;
                    }
                }

                TestModel(e);
            }

            TestModel(Epoch, true);
        }
    }
}
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

        public LogisticRegression(Dataset dataset, bool constInit = false, float initVal = 1f)
        {
            HighestResult = 0;
            HighestResultEpoch = 0;

            Data = dataset;
            Classes = dataset.classCount;
            Heatmap = new List<(int, int)>();
            Neurons = new Neuron[Classes];
            Output = new float[Classes];

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(Data.InputLenght, constInit, initVal);
            }
        }

        public void TrainModel(int epoch, float learningRate, float momentum = 0, int miniBatch = 1, bool validatewithTS = false)
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

            Trainer(momentum, validatewithTS);

            Plot.Graph(PlotData, LearningRate, MiniBatch, HighestResultEpoch);
        }

        private void TestModel(int epoch, bool validateWithTrainingSet, bool plot = false)
        {
            var Prediction = new float[Classes];
            var classcount = new int[Classes];

            var TestPlot = new double[Classes, Classes];
            var testx = new List<int>();
            var testy = new List<int>();

            var ValidationSet = (validateWithTrainingSet == true ? Data.TrainingData : Data.ValidationgData);

            if (plot)
            {
                var faults = new int[Classes];
                var correct = new int[Classes];
                var plt = new ScottPlot.Plot(600, 400);
                var defaultMarker = 10;
                Heatmap.Clear();

                foreach (var input in ValidationSet)
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
                    TestPlot[yPos, position]++;

                    testx.Add(yPos);
                    testy.Add(position);
                }

                /*
                for (int i = 0; i < TestPlot.GetLength(0); i++)
                {
                    for (int j = 0; j < TestPlot.GetLength(1); j++)
                    {
                        var multiplier = TestPlot[i, j] / classcount[i];
                        plt.PlotPoint(i, j, markerSize: 50 * multiplier, color: Color.Green);
                    }
                }
                */

                /*
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
                */

                //TestPlot[5, 5] = 0;

                //var intesi = ScottPlot.Tools.XYToIntensities(ScottPlot.Tools.IntensityMode.gaussian, testx.ToArray(), testy.ToArray(), 50, 50, 4);

                var plottables = plt.GetPlottables();

                for (int i = 0; i < TestPlot.GetLength(0); i++)
                {
                    for (int j = 0; j < TestPlot.GetLength(1); j++)
                    {
                        var multiplier = Math.Round(TestPlot[i, j] / classcount[i], 2);

                        plottables.Add(new ScottPlot.PlottableText(multiplier.ToString(), i, j,
                            color: Color.Black, fontName: "arial", fontSize: 15,
                            bold: (i == j ? true : false), label: "", alignment: ScottPlot.TextAlignment.middleCenter,
                            rotation: 0, frame: false, frameColor: Color.Green));
                    }
                }

                //plt.PlotHeatmap(intesi);

                plt.Grid(xSpacing: 1, ySpacing: 1);
                plt.SaveFig("HeatmapImage.png");
                Process.Start(new ProcessStartInfo("HeatmapImage.png") { UseShellExecute = true });
            }
            else
            {
                var Loss = 0f;
                var Accuracy = 0f;

                foreach (var input in ValidationSet)
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

                Acc = (float)Accuracy / ValidationSet.Length;

                if (Acc > HighestResult)
                {
                    HighestResult = Acc;
                    HighestResultEpoch = epoch;
                }

                PlotData[epoch, 0] = Loss / ValidationSet.Length;
                PlotData[epoch, 1] = Acc;
                Console.WriteLine("Epoch: " + epoch + " Acccuracy: " + Acc + " Loss: " + Loss / ValidationSet.Length);
            }
        }

        private void Trainer(float momentum, bool validateWithTrainingSet)
        {
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

                    //prediction
                    Output = activation.Activate(Neurons);

                    //var pos = Output.ToList().IndexOf(Output.Max());

                    //var pred = new float[Classes];

                    //pred[pos] = 1;

                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].Derivate = activation.Derivate(Output[i], input.TruthLabel[i]);
                        //Neurons[i].Derivate = activation.Derivate(pred[i], input.TruthLabel[i]);
                        Neurons[i].Backpropagate(input.Data);
                    }

                    BatchCount++;

                    if (BatchCount == MiniBatch)
                    {
                        for (int i = 0; i < Neurons.Length; i++)
                        {
                            Neurons[i].AdjustWeights(MiniBatch, LearningRate, momentum);
                        }

                        BatchCount = 0;
                    }
                }

                TestModel(e, validateWithTrainingSet, plot: false);
            }

            TestModel(Epoch, validateWithTrainingSet, plot: true);
        }
    }
}
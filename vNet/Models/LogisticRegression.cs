using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using vNet.Activations;
using vNet.LossFunctions;
using vNet.Regularization;

namespace vNet
{
    internal class LogisticRegression : ModelType
    {
        private float LearningRate, Momentum, HighestResult, Acc;
        private int MiniBatch, HighestResultEpoch, Epoch, StepDecay, DLT, DUT, Classes;
        private double[,] PlotData;
        private List<(int, int)> Heatmap;
        private float[] Output;
        private Activation activation;
        private Loss loss;
        private Dataset Data;
        private Neuron[] Neurons;

        public LogisticRegression(Dataset dataset, int DropoutLowerThreshold = 0, int DropoutUpperThreshold = 0, bool L2 = false, bool constInit = false, float initVal = 1f)
        {
            HighestResult = 0;
            HighestResultEpoch = 0;
            Data = dataset;
            Classes = dataset.classCount;
            Heatmap = new List<(int, int)>();
            Neurons = new Neuron[Classes];
            Output = new float[Classes];

            if (DropoutLowerThreshold != 0 & DropoutUpperThreshold != 0)
            {
                DLT = DropoutLowerThreshold;
                DUT = DropoutUpperThreshold;
                var temp = new List<int>();
                var interMidLayer = new int[dataset.InputLenght];

                for (int i = 0; i < dataset.TrainingData.Length; i++)
                {
                    for (int j = 0; j < dataset.InputLenght; j++)
                    {
                        interMidLayer[j] += dataset.TrainingData[i].Data[j] > 0 ? 1 : 0;
                    }
                }

                for (int i = 0; i < interMidLayer.Length; i++)
                {
                    if (interMidLayer[i] > DLT & interMidLayer[i] < DUT)
                    {
                        temp.Add(i);
                    }
                }

                var ConnectionPattern = temp.ToArray();

                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i] = new Neuron(ConnectionPattern, constInit, initVal, L2);
                }

                Data.ApplyConnectionMask(ConnectionPattern);
            }
            else
            {
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i] = new Neuron(Data.InputLenght, constInit, initVal, L2);
                }
            }
        }

        public void TrainModel(int epoch, float learningRate, int stepDecay = 0, float momentum = 0, int miniBatch = 1, bool validatewithTS = false)
        {
            StepDecay = stepDecay;
            Epoch = epoch;
            LearningRate = learningRate;
            Momentum = momentum;
            MiniBatch = miniBatch;
            PlotData = new double[Epoch, 3];

            if (MiniBatch == 0) { MiniBatch = Data.TrainingData.Length; }

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

            Console.WriteLine("-----Starting training-----\n" +
                "<-Parameters->\n" +
                "Epoch: {0}\n" +
                "Batchsize: {3}\n" +
                "Learningrate: {1}\n" +
                "Momentum: {4}\n" +
                "Dropout lower threshold: {2}\n" +
                "Dropout upper threshold: {5}\n",
                Epoch, LearningRate, DLT, MiniBatch, Momentum, DUT);

            Trainer(Momentum, validatewithTS);

            Plot.Graph(PlotData, LearningRate, MiniBatch, HighestResultEpoch);
        }

        private void TestModel(int epoch, bool validateWithTrainingSet, bool plot = false)
        {
            var Prediction = new int[Classes][];

            for (int i = 0; i < Prediction.Length; i++)
            {
                Prediction[i] = new int[Classes];
            }

            var classcount = new int[Classes];

            var TestPlot = new double[Classes, Classes];
            var testx = new List<int>();
            var testy = new List<int>();

            var Misclassified = new List<int>();

            var ValidationSet = (validateWithTrainingSet == true ? Data.TrainingData : Data.ValidationgData);

            if (plot)
            {
                var faults = new int[Classes];
                var correct = new int[Classes];

                var plt = new ScottPlot.Plot(800, 600);
                var pltMissclass = new ScottPlot.Plot(800, 600);
                bool imagesFull = false;

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

                    if (yPos != position)
                    {
                        Prediction[yPos][position]++;

                        pltMissclass.PlotBitmap(new Bitmap(Image.FromFile(input.Path)), yPos, position, alignment: ScottPlot.ImageAlignment.middleCenter);
                    }

                    Heatmap.Add((yPos, position));
                    TestPlot[yPos, position]++;

                    testx.Add(yPos);
                    testy.Add(position);
                }

                var plottables = plt.GetPlottables();

                for (int i = 0; i < TestPlot.GetLength(0); i++)
                {
                    for (int j = 0; j < TestPlot.GetLength(1); j++)
                    {
                        var multiplier = Math.Round(TestPlot[i, j] / classcount[i], 3);

                        if (multiplier > 0)
                        {
                            plottables.Add(new ScottPlot.PlottableText(multiplier.ToString(), i, j,
                            color: Color.Black, fontName: "arial", fontSize: 15,
                            bold: (i == j ? true : false), label: "", alignment: ScottPlot.TextAlignment.middleCenter,
                            rotation: 0, frame: false, frameColor: Color.Green));
                        }
                    }
                }

                var temp = classcount.Select(x => x.ToString()).ToArray();

                pltMissclass.Title("X: truth, Y: wrong prediction with image");
                //pltMissclass.XTicks(temp);
                pltMissclass.Grid(xSpacing: 1, ySpacing: 1);
                pltMissclass.SaveFig("missclass.png");
                Process.Start(new ProcessStartInfo("missclass.png") { UseShellExecute = true });

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

                    Loss += loss.Calculate(Output, input.TruthLabel);

                    // Convert output
                    int position = Output.ToList().IndexOf(Output.Max());
                    var yPos = input.TruthLabel.ToList().IndexOf(input.TruthLabel.Max());

                    Accuracy += position == yPos ? 1 : 0;
                }

                Acc = (float)Math.Round(Accuracy / ValidationSet.Length, 3);

                if (Acc > HighestResult)
                {
                    HighestResult = Acc;
                    HighestResultEpoch = epoch;
                }

                PlotData[epoch, 0] = Loss / ValidationSet.Length;
                PlotData[epoch, 1] = Acc;
                Console.WriteLine("E: " + epoch + " Loss: " + Loss / ValidationSet.Length + " LR: " + LearningRate + " Acc: " + Acc);
            }
        }

        private void Trainer(float momentum, bool validateWithTrainingSet)
        {
            int BatchCount = 0;

            int StepDecayCounter = 0;

            for (int e = 0; e < Epoch; e++)
            {
                //timer.Restart();
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

                    PlotData[e, 2] += loss.Calculate(Output, input.TruthLabel);

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

                PlotData[e, 2] /= Data.TrainingData.Length;
                TestModel(e, validateWithTrainingSet, plot: false);

                StepDecayCounter++;

                if (StepDecayCounter == StepDecay)
                {
                    LearningRate *= .75f;
                    StepDecayCounter = 0;
                }
            }

            TestModel(Epoch, validateWithTrainingSet: false, plot: true);
        }
    }
}
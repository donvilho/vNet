using Microsoft.VisualBasic.FileIO;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using vNet.Activations;
using vNet.LossFunctions;

namespace vNet
{
    public class Trainer
    {
        public Neuron[] Neurons { get; set; }
        public int Classes { get; set; }
        public double[] Output { get; set; }
        public Activation activation { get; set; }
        public Loss loss { get; set; }

        public double HighestResult;

        private int[] Mask;

        private int Epoch, Batch, StepDecay, HighestEpoch;
        private double Lr, Momentum, InitLr, BestLoss;
        private bool L2;
        private double[,] PlotData;

        private Dataset Data;

        public Trainer(Dataset _dataset)
        {
            //Data = Utils.DatasetFromBinary(path);
            Data = _dataset;
            Classes = Data.classCount;
            Neurons = new Neuron[Classes];
            Output = new double[Classes];
            BestLoss = 0f;

            Mask = null;

            if (Data.connectionMask != null)
            {
                Mask = Data.connectionMask;
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i] = new Neuron(Data.connectionMask);
                }
            }
            else
            {
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i] = new Neuron(Data.InputLenght);
                }
            }

            if (Classes > 2)
            {
                activation = new Softmax();
                loss = new CrossEntropy();
            }
            else
            {
                activation = new Sigmoid();
                loss = new MSE();
            }
        }

        public void Run(string path)
        {
            var attrib = File.GetAttributes(path);

            if ((attrib & FileAttributes.Directory) == FileAttributes.Directory)
            {
                var files = Directory.GetFiles(path);
                for (int i = 0; i < files.Length; i++)
                {
                    var file = Utils.ImageToArray(files[i]);

                    if (Mask != null)
                    {
                        file = Utils.ApplyConnectionMask(Mask, file);
                    }

                    //forward
                    for (int j = 0; j < Neurons.Length; j++)
                    {
                        Neurons[j].ForwardCalculation(file);
                    }

                    //activate
                    var Output = activation.Activate(Neurons);

                    var prediction = Output.ToList().IndexOf(Output.Max());

                    FileSystem.RenameFile(files[i], prediction + Path.GetExtension(files[i]));

                    Console.WriteLine(files[i] + " is labeled as: " + prediction);
                }
            }
            else
            {
                var file = Utils.ImageToArray(path);
                //forward
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i].ForwardCalculation(file);
                }

                //activate
                var Output = activation.Activate(Neurons);

                Console.WriteLine(path + " is labeled as: " + Output.ToList().IndexOf(Output.Max()));
            }
        }

        public void Init(double lr, int batch, int epoch, double momentum, int stepDecay, bool l2)
        {
            Epoch = epoch;
            Batch = batch;
            StepDecay = stepDecay;
            Lr = lr;
            InitLr = lr;
            Momentum = momentum;
            L2 = l2;
            PlotData = new double[epoch, 4];
            HighestResult = 0;
            HighestEpoch = 0;
            if (batch == 0) { Batch = Data.TrainingData.Length; }
        }

        public void Train(bool print, bool DevSet)
        {
            if (Vector.IsHardwareAccelerated) Console.WriteLine("SIMD enabled");
            var dataset = DevSet == true ? Data.DevSet : Data.TrainingData;
            Batch = Batch == 0 ? dataset.Length : Batch;
            int BatchCount = 0;
            int StepDecayCounter = 0;
            var epochTimer = new Stopwatch();
            var totalTimer = new Stopwatch();
            var avgTime = 0f;
            totalTimer.Start();

            for (int e = 0; e < Epoch; e++)
            {
                epochTimer.Restart();
                Data.Shuffle(dataset);

                var trainingAccuracy = 0f;

                //Training loop
                foreach (var input in dataset)
                {
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].ForwardCalculation(input.Data);
                    }

                    //prediction
                    var Output = activation.Activate(Neurons);

                    trainingAccuracy += activation.Compare(Output, input.TruthLabel);

                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].Derivate = activation.Derivate(Output[i], input.TruthLabel[i]);
                        Neurons[i].Backpropagate(input.Data);
                    }
                    BatchCount++;

                    if (BatchCount == Batch)
                    {
                        for (int i = 0; i < Neurons.Length; i++)
                        {
                            Neurons[i].AdjustWeights(Batch, Lr, Momentum, L2);
                        }

                        BatchCount = 0;
                    }
                }

                if (BatchCount > 0)
                {
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].AdjustWeights(BatchCount, Lr, Momentum, L2);
                    }
                    BatchCount = 0;
                }

                var result = TestModel(Data);

                PlotData[e, 2] = trainingAccuracy / (dataset.Length - 1);
                PlotData[e, 0] = result.Item1;
                PlotData[e, 1] = result.Item2;
                PlotData[e, 3] = totalTimer.Elapsed.TotalSeconds;

                avgTime += epochTimer.ElapsedMilliseconds;

                if (print)
                {
                    Console.WriteLine("E:" + e + " -- Loss: " + result.Item1 + " -- Training: " + Math.Round(PlotData[e, 2], 3) + " -- Validation: " + result.Item2);
                }

                if (result.Item2 > HighestResult)
                {
                    HighestResult = result.Item2;
                    HighestEpoch = e;
                }

                if (BestLoss == 0)
                {
                    BestLoss = result.Item1;
                }
                else if (result.Item1 < BestLoss)
                {
                    BestLoss = result.Item1;
                }

                /*
                else if (HighestResult - result.Item2 > 0.05)
                {
                    return;
                    //return (PlotData, HighestEpoch, initLr, Batch, 0f, L2, Momentum);
                }
                */

                StepDecayCounter++;

                if (StepDecay > 0 & StepDecayCounter == StepDecay)
                {
                    Lr *= .95f;
                    StepDecayCounter = 0;
                }
            }

            totalTimer.Stop();
            Console.WriteLine("Lr: " + InitLr + " Batch: " + Batch + " -- Average epoch time: " + avgTime / Epoch + " ms --  Total time was: " + Math.Round(totalTimer.Elapsed.TotalSeconds, 3) + " s");

            //return (PlotData, HighestResultEpoch, initLr, Batch, HighestResult, L2, Momentum);
        }

        public void FocusedTraining(Dataset Data)
        {
            var FNeurons = (Neuron[])Neurons.Clone();
            int BatchCount = 0;
            int StepDecayCounter = 0;
            Lr = 0.01f;
            Momentum = 0;
            bool train = true;
            Console.WriteLine("Focused Training");
            while (train)
            {
                for (int e = 0; e < Epoch; e++)
                {
                    Data.Shuffle(Data.TrainingData);
                    var trainingAccuracy = 0f;
                    //Training loop
                    foreach (var input in Data.TrainingData)
                    {
                        for (int i = 0; i < FNeurons.Length; i++)
                        {
                            FNeurons[i].ForwardCalculation(input.Data);
                        }

                        //prediction
                        var Output = activation.Activate(FNeurons);

                        trainingAccuracy += Output.ToList().IndexOf(Output.Max()) == input.TruthLabel.ToList().IndexOf(input.TruthLabel.Max()) ? 1 : 0;

                        for (int i = 0; i < FNeurons.Length; i++)
                        {
                            FNeurons[i].Derivate = activation.Derivate(Output[i], input.TruthLabel[i]);
                            FNeurons[i].Backpropagate(input.Data);
                        }

                        BatchCount++;

                        if (BatchCount == Batch)
                        {
                            for (int i = 0; i < FNeurons.Length; i++)
                            {
                                FNeurons[i].AdjustWeights(Batch, Lr, Momentum, L2);
                            }

                            BatchCount = 0;
                        }
                    }

                    for (int i = 0; i < FNeurons.Length; i++)
                    {
                        FNeurons[i].AdjustWeights(BatchCount, Lr, Momentum, L2);
                    }
                    BatchCount = 0;

                    var result = TestModel(Data);

                    PlotData[e, 2] = trainingAccuracy / Data.TrainingData.Length;
                    PlotData[e, 0] = result.Item1;
                    PlotData[e, 1] = result.Item2;

                    Console.WriteLine("Loss: " + result.Item1 + " Acc: " + result.Item2 + " Lr: " + Lr);

                    if (result.Item1 > BestLoss)
                    {
                        train = false;
                    }
                    else if (result.Item1 < BestLoss)
                    {
                        BestLoss = result.Item1;
                    }

                    if (result.Item2 > HighestResult)
                    {
                        HighestResult = result.Item2;
                        HighestEpoch = e;
                    }
                    else if (HighestResult - result.Item2 > 0.05)
                    {
                        Console.WriteLine("break");
                        train = false;
                    }

                    StepDecayCounter++;

                    if (StepDecayCounter == 5)
                    {
                        Lr *= .95f;
                        StepDecayCounter = 0;
                    }
                }
            }

            Console.WriteLine("Loss: " + BestLoss + " Acc: " + HighestResult + " Lr: " + Lr);

            Console.ReadKey();
        }

        private (double, double) TestModel(Dataset Data)
        {
            var Loss = 0d;
            var Accuracy = 0d;

            foreach (var input in Data.ValidationData)
            {
                //forward
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i].ForwardCalculation(input.Data);
                }

                //activate
                var Output = activation.Activate(Neurons);

                Loss += loss.Calculate(Output, input.TruthLabel);

                // Convert output
                /*
                int position = Output.ToList().IndexOf(Output.Max());
                var yPos = input.TruthLabel.ToList().IndexOf(input.TruthLabel.Max());

                Accuracy += position == yPos ? 1 : 0;
                */

                Accuracy += activation.Compare(Output, input.TruthLabel);
            }

            Accuracy = (double)Math.Round(Accuracy / Data.ValidationData.Length, 3);
            Loss /= Data.ValidationData.Length;
            return (Loss, Accuracy);
        }

        public void PlotModel()
        {
            var classcount = new int[Classes];
            var TestPlot = new double[Classes, Classes];
            var Prediction = new int[Classes][];
            var Heatmap = new List<(int, int)>();
            var testx = new List<int>();
            var testy = new List<int>();
            var faults = new int[Classes];
            var correct = new int[Classes];
            var Misclassified = new List<int>();

            for (int i = 0; i < Prediction.Length; i++)
            {
                Prediction[i] = new int[Classes];
            }

            var plt = new ScottPlot.Plot(800, 600);
            var pltMissclass = new ScottPlot.Plot(800, 600);

            Heatmap.Clear();

            foreach (var input in Data.ValidationData)
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

            pltMissclass.Title("X: truth, Y: prediction");
            //pltMissclass.XTicks(temp);
            pltMissclass.Grid(xSpacing: 1, ySpacing: 1);
            pltMissclass.SaveFig("missclass.png");
            Process.Start(new ProcessStartInfo("missclass.png") { UseShellExecute = true });

            plt.Grid(xSpacing: 1, ySpacing: 1);
            plt.SaveFig("HeatmapImage.png");
            Process.Start(new ProcessStartInfo("HeatmapImage.png") { UseShellExecute = true });
        }

        public (double, double[,], double, double, int, int, bool) GetResult()
        {
            return (HighestResult, PlotData, InitLr, Momentum, Batch, HighestEpoch, L2);
        }

        public string GetValue()
        {
            return "Lr: " + InitLr + " Batch: " + Batch + " Accuracy: " + HighestResult + " E: " + HighestEpoch;
        }
    }
}
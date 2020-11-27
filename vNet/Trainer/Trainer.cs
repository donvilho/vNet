using Microsoft.VisualBasic.FileIO;
using System;
using System.IO;
using System.Linq;
using System.Threading;
using vNet.Activations;
using vNet.LossFunctions;

namespace vNet
{
    public class Trainer
    {
        public Neuron[] Neurons { get; set; }
        public int Classes { get; set; }
        public float[] Output { get; set; }
        public Activation activation { get; set; }
        public Loss loss { get; set; }
        private string[] Labels;

        public float HighestResult;

        private int Epoch, Batch, StepDecay, HighestEpoch;
        private float Lr, Momentum, InitLr, BestLoss;
        private bool L2;
        private double[,] PlotData;

        public Trainer(Dataset Data)
        {
            //Console.WriteLine("new trainer");
            Classes = Data.classCount;
            Neurons = new Neuron[Classes];
            Output = new float[Classes];
            BestLoss = 0f;

            if (Data.connectionMask != null)
            {
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i] = new Neuron(Data.connectionMask, 0);
                }
            }
            else
            {
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i] = new Neuron(Data.InputLenght, 0);
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
                loss = new CrossEntropy();
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
                    //forward
                    for (int j = 0; j < Neurons.Length; j++)
                    {
                        Neurons[j].ForwardCalculation(file);
                    }

                    //activate
                    var Output = activation.Activate(Neurons);

                    var prediction = Output.ToList().IndexOf(Output.Max());

                    FileSystem.RenameFile(files[i], prediction + "__" + Path.GetFileName(files[i]));

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

        public void Init(float lr, int batch, int epoch, float momentum, int stepDecay, bool l2)
        {
            Epoch = epoch;
            Batch = batch;
            StepDecay = stepDecay;
            Lr = lr;
            InitLr = lr;
            Momentum = momentum;
            L2 = l2;
            PlotData = new double[epoch, 3];
            HighestResult = 0;
            HighestEpoch = 0;
        }

        public void Train(Dataset dataset, bool print, bool DevSet)
        {
            var Data = DevSet == true ? dataset.DevSet : dataset.TrainingData;

            Batch = Batch == 0 ? Data.Length : Batch;
            int BatchCount = 0;
            int StepDecayCounter = 0;
            Console.WriteLine(Thread.CurrentThread.ManagedThreadId + " ");

            for (int e = 0; e < Epoch; e++)
            {
                dataset.Shuffle(Data);

                var trainingAccuracy = 0f;

                //Training loop
                foreach (var input in Data)
                {
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i].ForwardCalculation(input.Data);
                    }

                    //prediction
                    var Output = activation.Activate(Neurons);

                    trainingAccuracy += Output.ToList().IndexOf(Output.Max()) == input.TruthLabel.ToList().IndexOf(input.TruthLabel.Max()) ? 1 : 0;

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

                var result = TestModel(dataset);

                PlotData[e, 2] = trainingAccuracy / Data.Length;
                PlotData[e, 0] = result.Item1;
                PlotData[e, 1] = result.Item2;

                if (print)
                {
                    Console.WriteLine("E: " + e + " Loss: " + result.Item1 + " Test acc: " + result.Item2 + " Training acc: " + Math.Round(PlotData[e, 2], 3) + "  Lr: " + Lr);
                }

                if (BestLoss == 0)
                {
                    BestLoss = result.Item1;
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
                    HighestResult = 0;
                    return;
                    //return (PlotData, HighestEpoch, initLr, Batch, 0f, L2, Momentum);
                }

                StepDecayCounter++;

                if (StepDecayCounter == StepDecay)
                {
                    Lr *= .95f;
                    StepDecayCounter = 0;
                }
            }

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

        private (float, float) TestModel(Dataset Data)
        {
            var Loss = 0f;
            var Accuracy = 0f;

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
                int position = Output.ToList().IndexOf(Output.Max());
                var yPos = input.TruthLabel.ToList().IndexOf(input.TruthLabel.Max());

                Accuracy += position == yPos ? 1 : 0;
            }

            Accuracy = (float)Math.Round(Accuracy / Data.ValidationData.Length, 3);
            Loss /= Data.ValidationData.Length;
            return (Loss, Accuracy);
        }

        public (float, double[,], float, float, int, int, bool) GetResult()
        {
            return (HighestResult, PlotData, InitLr, Momentum, Batch, HighestEpoch, L2);
        }
    }
}
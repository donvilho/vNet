using System;
using System.Linq;
using System.Threading;
using vNet.Activations;
using vNet.LossFunctions;

namespace vNet
{
    internal class Trainer
    {
        public Neuron[] Neurons { get; set; }
        public int Classes { get; set; }
        public float[] Output { get; set; }
        public Activation activation { get; set; }
        public Loss loss { get; set; }

        public Trainer(Dataset dataset, int DropoutLowerThreshold = 0, int DropoutUpperThreshold = 0, bool L2 = false, bool constInit = false, float initVal = 1f)
        {
            Classes = dataset.classCount;
            Neurons = new Neuron[Classes];
            Output = new float[Classes];

            if (dataset.connectionMask != null)
            {
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i] = new Neuron(dataset.connectionMask, constInit, initVal, L2);
                }
            }
            else
            {
                for (int i = 0; i < Neurons.Length; i++)
                {
                    Neurons[i] = new Neuron(dataset.InputLenght, constInit, initVal, L2);
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

        public (double[,], int, float, int) Train(Dataset Data, int epoch, float lr, int batch, float momentum, int stepDecay)
        {
            if (batch == 0) batch = Data.TrainingData.Length;

            float initLr = lr;
            var PlotData = new double[epoch, 3];
            float HighestResult = 0;
            int HighestResultEpoch = 0;
            int BatchCount = 0;
            int StepDecayCounter = 0;

            for (int e = 0; e < epoch; e++)
            {
                Console.WriteLine(Thread.CurrentThread.ManagedThreadId + " " + e);
                Data.Shuffle(Data.TrainingData);
                var trainingAccuracy = 0f;
                //Training loop
                foreach (var input in Data.TrainingData)
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

                    if (BatchCount == batch)
                    {
                        for (int i = 0; i < Neurons.Length; i++)
                        {
                            Neurons[i].AdjustWeights(batch, lr, momentum);
                        }

                        BatchCount = 0;
                    }
                }

                PlotData[e, 2] = trainingAccuracy / Data.TrainingData.Length;
                var result = TestModel(Data);

                PlotData[e, 0] = result.Item1;
                PlotData[e, 1] = result.Item2;

                if (result.Item2 > HighestResult)
                {
                    HighestResult = result.Item2;
                    HighestResultEpoch = e;
                }

                StepDecayCounter++;

                if (StepDecayCounter == stepDecay)
                {
                    lr *= .75f;
                    StepDecayCounter = 0;
                }
            }

            return (PlotData, HighestResultEpoch, initLr, batch);
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
    }
}
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

        public LogisticRegression(Dataset dataset, int DropoutLowerThreshold = 0, int DropoutUpperThreshold = 0, bool L2 = false, bool constInit = false, float initVal = 1f)
        {
            Data = dataset;

            // Connection mask

            if (DropoutLowerThreshold != 0 & DropoutUpperThreshold != 0)
            {
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
                    if (interMidLayer[i] > DropoutLowerThreshold & interMidLayer[i] < DropoutUpperThreshold)
                    {
                        temp.Add(i);
                    }
                }

                Data.ApplyConnectionMask(temp.ToArray());
            }
        }

        public void TrainModel(int epoch, float learningRate, int stepDecay = 0, float momentum = 0, int miniBatch = 1)
        {
            if (miniBatch == 0) { miniBatch = Data.TrainingData.Length; }

            Console.WriteLine("-----Starting training-----\n" +
                "<-Parameters->\n" +
                "Epoch: {0}\n" +
                "Batchsize: {3}\n" +
                "Learningrate: {1}\n" +
                "Momentum: {4}\n",
                epoch, learningRate, miniBatch, momentum);

            var trainer = new Trainer(Data);
            var result = trainer.Train(Data, epoch, learningRate, miniBatch, momentum, stepDecay);

            Plot.Graph(result.Item1, learningRate, miniBatch, result.Item2);
        }

        public void MultiTraining(TrainingSetup setup, int epoch, float learningRate, int stepDecay = 0, float momentum = 0, int miniBatch = 1, bool validatewithTS = false)
        {
            if (miniBatch == 0) { miniBatch = Data.TrainingData.Length; }

            var cBag = new ConcurrentBag<(double[,], int, float, int)>();

            var trainer = new ThreadLocal<Trainer>(() => new Trainer(Data));
            Parallel.For(0, setup.learningrates.Length, i =>
             {
                 for (int j = 0; j < setup.batches.Length; j++)
                 {
                     Console.WriteLine(Thread.CurrentThread.ManagedThreadId + " started " + setup.learningrates[i] + " - " + setup.batches[j]);
                     cBag.Add(trainer.Value.Train(Data, epoch, setup.learningrates[i], setup.batches[j], momentum, stepDecay));
                     Console.WriteLine(Thread.CurrentThread.ManagedThreadId + " done");
                 };
             });

            Console.WriteLine();

            Plot.GraphList(cBag.ToList());
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="epoch"></param>
        /// <param name="validateWithTrainingSet"></param>
        /// <returns>(Loss,Accuracy)</returns>

        /*
        private (double[,], int) Trainer2BU(int epoch, float lr, int batch, float momentum, int stepDecay)
        {
            var PlotData = new double[epoch, 3];
            float HighestResult = 0;
            int HighestResultEpoch = 0;
            int BatchCount = 0;
            int StepDecayCounter = 0;

            for (int e = 0; e < epoch; e++)
            {
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
                    Output = activation.Activate(Neurons);

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
                var result = TestModel();

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

            return (PlotData, HighestResultEpoch);
        }
        */

        private void TrainerBU(float momentum, bool validateWithTrainingSet)
        {
            /*
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
            */
        }
    }
}
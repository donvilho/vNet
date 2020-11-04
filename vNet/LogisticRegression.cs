using ScottPlot;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace vNet
{
    class LogisticRegression : IModel
    {
        private int Epoch;
        private float LearningRate;
        private int MiniBatch;
        private double[,] PlotData;
        private List<(int,int)> Heatmap;
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

                    if(BatchCount == MiniBatch)
                    {
                        Net.UpdateWeights(MiniBatch, LearningRate);
                        BatchCount = 0;
                        
                    }
   
                }
               
                TestNet(Dataset.ValidationgData, e);
                
            }

            TestNet(Dataset.ValidationgData,Epoch,true);

            Plot.Graph(PlotData, LearningRate,MiniBatch);
     
        }


        private void TestNet(Input[] Data, int epoch, bool plot=false)
        {
            
            var trainer = new Trainer(Net);
            var TestError = 0f;
            var Accuracy = 0f;
            var classcount = new int[Classes];
            Heatmap.Clear();

            foreach (var input in Data)
            {
                var result = trainer.Test(input);

                TestError += result.Item1;
                if (result.Item2) { Accuracy++; }

                var yPos = 0;

                for (int i = 0; i < input.TruthLabel.Length; i++)
                {
                    if(input.Data[i] == 1)
                    {
                        yPos = i;
                        classcount[i]++;
                    }
                }

               
                    Heatmap.Add((yPos,result.Item3));
  
            }

            if (!plot)
            {
                PlotData[epoch, 0] = TestError / Data.Length;
                PlotData[epoch, 1] = Accuracy / Data.Length;

            }

            //Console.WriteLine(TestError / TestData.Length);
            Console.WriteLine("Epoch: " + epoch + " Acccuracy: " + Accuracy / Data.Length + " Error: " + TestError / Data.Length);

            if ((Accuracy / Data.Length > 0.99) | plot)
            {

                var faults = new int[Classes];
                var correct = new int[Classes];
                var plt = new ScottPlot.Plot(600, 400);

                var defaultMarker = 10;

                foreach (var item in Heatmap)
                {
                    if(item.Item1 == item.Item2)
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
        }


        private float Dot(float[] a, float[] b)
        {
            var temp = 0f;
            for (int i = 0; i < b.Length; i++)
            {
                temp += a[i] * b[i];
            }
            return temp;

        }

        private float ActivateNode(float A)
        {
            return (float)(1 / (1 + Math.Exp(-A)));
        }

        public void Shuffle((float[],float[],string)[] Array)
        {
            Console.WriteLine("Shuffle start");

            var rand = new Random();
            for (int Count = Array.Length - 1; Count > 1; Count--)
            {
                int i = rand.Next(Count + 1);
                var value = Array[i];
                Array[i] = Array[Count];
                Array[Count] = value;
            }
            Console.WriteLine("Shuffle finish");
        }

    }
}
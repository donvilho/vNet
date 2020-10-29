using ScottPlot;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace vNet
{
    class LogisticRegression : IModel
    {
        public  float[] Neurons, Error, Derivate;
        public float[][] Weights;
        public float[][][] WeightCache;
        public float[] Bias, BiasCache;
        private  int Epoch;
        private  float LearningRate;
        private  int MiniBatch;

        //private readonly Network Net;

        public LogisticRegression(int epoch, float learningrate, int minibatch = 32)
        {
            Epoch = epoch;
            LearningRate = learningrate;
            MiniBatch = minibatch;
            

        }

        public void TestModel()
        {
            throw new NotImplementedException();
        }

        public unsafe void TrainModel(string path, bool plot = false)
        {
           
            var rand = new Random();
            var Datasets = Utils.DataArrayCreator(path);

            var TestData = Datasets.Item2;

            for (int Count = Datasets.Item1.Length - 1; Count > 1; Count--)
            {
                int i = rand.Next(Count + 1);
                var value = Datasets.Item1[i];
                Datasets.Item1[i] = Datasets.Item1[Count];
                Datasets.Item1[Count] = value;
            }

            var Batches = Utils.SplitToMiniBatch(Datasets.Item1, MiniBatch);

            var neuronCount = Datasets.Item2[0].Item2.Length;
            var inputLenght = Datasets.Item2[0].Item1.Length;

            Neurons = new float[neuronCount];
            Error = new float[neuronCount];
            Derivate = new float[neuronCount];

            Bias = Utils.Generate_Vector(neuronCount);
            BiasCache = new float[neuronCount];

            Weights = new float[neuronCount][];
            WeightCache = new float[MiniBatch][][];

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Utils.Generate_Vector(inputLenght, 0.0001, 0.0009);
                WeightCache[i] = new float[neuronCount][];
                for (int j = 0; j < WeightCache[i].Length; j++)
                {
                    WeightCache[i][j] = new float[inputLenght];
                }
            }
         

            var BCache = new ConcurrentBag<float[]>();
            var WCache = new ConcurrentBag<float[][]>();


            double[,] PlotData = new double[Epoch, 2];

            
            for (int e = 0; e < Epoch; e++)
            {

                Console.WriteLine(e);
                var EpochError = new ConcurrentBag<float>();

                //shuffle
                
                for (int Count = Batches.Length - 1; Count > 1; Count--)
                {
                    int i = rand.Next(Count + 1);
                    var value = Batches[i];
                    Batches[i] = Batches[Count];
                    Batches[Count] = value;
                }
                
                foreach (var batch in Batches)
                {
                   
                    Parallel.For(0, batch.Length, i =>
                    {                   
                       Train(batch[i],);
                    });

                  

                }



                var TestError = 0f;
                var Accuracy = 0f;

                
                for (int Count = TestData.Length - 1; Count > 1; Count--)
                {
                    int i = rand.Next(Count + 1);
                    var value = TestData[i];
                    TestData[i] = TestData[Count];
                    TestData[Count] = value;
                }


                //Console.WriteLine("Test Start");
                /*
                for (int input = 0; input < TestData.Length; input++)
                {

                    //Forward
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i] = Bias[i] + Dot(Weights[i], TestData[input].Item1);
                    }

                    var Loss = 0f;
                    //Calc EXP SUM
                    var ExpSum = 0f;
                    foreach (var neuron in Neurons)
                    {
                        ExpSum += (float)Math.Exp(neuron);
                    }

                    //CalcError/activate
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Error[i] = (float)Math.Exp(Neurons[i]) / ExpSum;

                        //loss - softmax

                        if(float.IsInfinity(TestData[input].Item2[i] * (float)Math.Log(Error[i])))
                        {
                            Console.WriteLine("infinity");
                            Console.WriteLine(TestData[input].Item2[i]);
                            Console.WriteLine((float)Math.Log(Error[i]));
                        }
                        else
                        {
                            Loss += TestData[input].Item2[i] * (float)Math.Log(Error[i]);
                        }

                        

                        Error[i] = (float)Math.Round(Error[i]);

                    }
                    Loss = -Loss;
                    TestError += Loss;

                    if (Error.SequenceEqual(TestData[input].Item2)) { Accuracy++; }

                }
                */
          
                //Console.WriteLine("Test Finish");

                PlotData[e, 0] = TestError / TestData.Length;
                PlotData[e, 1] = Accuracy / TestData.Length;

                Console.WriteLine(TestError);
                Console.WriteLine(Accuracy);
                //Console.WriteLine("accuracy: "+Accuracy/Dataset.ValidationgData.Length);


            }

            Plot.Graph(PlotData);
            

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

        /*
        public float Forward((float[], float[], string) input)
        {
            
            var ExpSum = 0f;

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = Bias[i] + Utils.Dot(Weights[i], input.Item1);
                //Calc EXP SUM
                ExpSum += (float)Math.Exp(Neurons[i]);
            }

            return ExpSum;
            
        }
        */

        /*
        public float Backward((float[], float[], string) input, float ExpSum, bool ParallelDegree = false)
        {
            var Loss = 0f;

            switch (ParallelDegree)
            {
                case false:
                    for (int i = 0; i < Derivate.Length; i++)
                    {
                        //CalcError/activate

                        Error[i] = (float)Math.Exp(Neurons[i]) / ExpSum;
                        Loss += input.Item2[i] * (float)Math.Log(Error[i]);

                        //Loss += input.Item2[i] * (float)Math.Log(Math.Exp(Neurons[i]) / ExpSum);
                        //CalcDerivates
                        //D-A
                        Derivate[i] = Error[i] - input.Item2[i];
                        //D-Z
                        Derivate[i] *= Error[i] * (1 - Error[i]);

                        for (int j = 0; j < WeightCache[i].Length; j++)
                        {
                            //D-W
                            WeightCache[i][j] += input.Item1[j] * Derivate[i];
                            //D-B
                            BiasCache[i] += Bias[i] * Derivate[i];
                        }
                    }
                    break;

                case true:

                    void Kernel(int i)
                    {
                        Error[i] = (float)Math.Exp(Neurons[i]) / ExpSum;
                        Loss += input.Item2[i] * (float)Math.Log(Error[i]);


                        Derivate[i] = (Error[i] - input.Item2[i]) * Error[i] * (1 - Error[i]);
                        //Derivate[i] *= Error[i] * (1 - Error[i]);

                        for (int j = 0; j < WeightCache[i].Length; j++)
                        {
                            //D-W
                            WeightCache[i][j] += input.Item1[j] * Derivate[i];
                            //D-B
                            BiasCache[i] += Bias[i] * Derivate[i];
                        }
                    }

                    Parallel.For(0, Derivate.Length, Kernel);
                    break;
            }
                
            return Loss;
        }
        */
        public void Train((float[], float[], string) input)
        {


            var ExpSum = 0f;
            var Loss = 0f;

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = Bias[i] + Utils.Dot(Weights[i], input.Item1);
                //Calc EXP SUM
                ExpSum += (float)Math.Exp(Neurons[i]);
            }



            for (int i = 0; i < Derivate.Length; i++)
            {
                //CalcError/activate

                Error[i] = (float)Math.Exp(Neurons[i]) / ExpSum;
                Loss += input.Item2[i] * (float)Math.Log(Error[i]);

                //Loss += input.Item2[i] * (float)Math.Log(Math.Exp(Neurons[i]) / ExpSum);
                //CalcDerivates
                //D-A
                Derivate[i] = Error[i] - input.Item2[i];
                //D-Z
                Derivate[i] *= Error[i] * (1 - Error[i]);



                for (int j = 0; j < WeightCache[i].Length; j++)
                {
                    //D-W
                    WeightCache[i][j] += input.Item1[j] * Derivate[i];
                    //D-B
                    BiasCache[i] += Bias[i] * Derivate[i];
                }
            }
            //return (WeightCache,BiasCache, Loss);
        }
    }
}
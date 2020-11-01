using ScottPlot;
using System;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace vNet
{
    class LogisticRegression : IModel
    {
        private float[] Error, Neurons, Derivate;
        private float[][] Weights, WeightCache;
        private float[] Bias, BiasCache;

        private readonly int Epoch;
        private readonly float LearningRate;
        private readonly int MiniBatch;

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


            var TrainingData = Datasets.Item1;
            var TestData = Datasets.Item2;

            var neuronCount = TrainingData[0].Item2.Length;
            var inputLenght = TrainingData[0].Item1.Length;
            
            
            Derivate = new float[neuronCount];
            Error = new float[neuronCount];
            Neurons = new float[neuronCount];

            Weights = new float[neuronCount][];
            WeightCache = new float[neuronCount][];

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Utils.Generate_Vector(inputLenght, 0.0001, 0.0009);
                WeightCache[i] = new float[inputLenght];
            }

            Bias = Utils.Generate_Vector(inputLenght);
            BiasCache = new float[inputLenght];
           

            /*
            Derivate = new float[dataset.TrainingData[0].Y.Length];
            Error = new float[dataset.TrainingData[0].Y.Length];
            Neurons = new float[dataset.TrainingData[0].Y.Length];

            Weights = new float[dataset.TrainingData[0].Y.Length][];
            WeightCache = new float[dataset.TrainingData[0].Y.Length][];

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Utils.Generate_Vector(dataset.InputLenght, 0.001, 0.009);
                WeightCache[i] = new float[dataset.InputLenght];
            }

            Bias = Utils.Generate_Vector(dataset.InputLenght);
            BiasCache = new float[dataset.InputLenght];
            */

            double[,] PlotData = new double[Epoch, 2];

            var BatchCount = 0;
            
            
            for (int e = 0; e < Epoch; e++)
            {

                Console.WriteLine(e);
                var EpochError = 0f;

                //Console.WriteLine("Start Training");

                for (int Count = TrainingData.Length - 1; Count > 1; Count--)
                {
                    int i = rand.Next(Count + 1);
                    var value = TrainingData[i];
                    TrainingData[i] = TrainingData[Count];
                    TrainingData[Count] = value;
                }

                foreach (var input in TrainingData)
                {
                    //Forward
                    //var ExpSum = 0f;
                    var Loss = 0f;

                    /*
                    var ExpSum = 0f;
                    
                    for (int i = 0; i < Neurons.Length; i++)
                    {
                        Neurons[i] = Bias[i] + Utils.Dot(Weights[i], input.Item1);
                        //Calc EXP SUM
                        ExpSum += (float)Math.Exp(Neurons[i]);
                    }  
                    */

                    var ExpSum = Forward(input);


                    /*
                    for (int i = 0; i < Derivate.Length; i++)
                    {
                        //CalcError/activate
                        Error[i] = (float)Math.Exp(Neurons[i]) / ExpSum;
                        Loss += input.Item2[i] * (float)Math.Log(Error[i]);
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
                    */

                    Loss = Backward(input,ExpSum);
                    Loss = -Loss;
                    EpochError += Loss;

                    BatchCount++;

                    if (BatchCount == MiniBatch)
                    {
                      

                        for (int i = 0; i < Weights.Length; i++)
                        {
                            Bias[i] -= (BiasCache[i] / MiniBatch) * LearningRate;
                            BiasCache[i] = 0;
                            for (int j = 0; j < Weights[i].Length; j++)
                            {
                              
                               Weights[i][j] -= (WeightCache[i][j] / MiniBatch) * LearningRate;
                               WeightCache[i][j] = 0;
                            }

                            BatchCount = 0;
                        }
                    }
                }

                //Console.WriteLine("Training finish");


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
                        Error[i] = (float)Utils.exp1(Neurons[i]) / ExpSum;

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


        public float Forward((float[], float[], string) input)
        {
            var ExpSum = 0f;

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = Bias[i] + Utils.Dot(Weights[i], input.Item1);
                //Calc EXP SUM
                ExpSum += (float)Utils.exp1(Neurons[i]);
            }

            return ExpSum;
        }

        public float Backward((float[], float[], string) input, float ExpSum, bool ParallelDegree = false)
        {
            var Loss = 0f;

            switch (ParallelDegree)
            {
                case false:
                    for (int i = 0; i < Derivate.Length; i++)
                    {
                        //CalcError/activate

                        Error[i] = (float)Utils.exp1(Neurons[i]) / ExpSum;
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
                        Error[i] = (float)Utils.exp1(Neurons[i]) / ExpSum;
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

    }
}
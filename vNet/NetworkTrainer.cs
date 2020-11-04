using System;
using System.Collections.Generic;
using System.Linq;

namespace vNet
{
    internal enum ModelType
    {
        Linear,
        Logistic,
    }

    internal enum Activator
    {
        Sigmoid,
        Relu,
        None
    }

    internal enum CostFunction
    {
        MAE,
        MSE,
        CEntropy,
        Logistic,
    }

    internal class NetworkTrainer
    {
        private float[][] Neurons;
        private float[][] NeuronDerivate;
        private float[][] NeuronsBackprop;

        private float[][] Bias;
        private float[][] BiasCache;

        private float[][][] Weights;
        private float[][][] WeightCache;

        private Activator[] Activations;

        private Type Type;


        public NetworkTrainer(int inputSize, List<(int, Activator)> Layers)
        {
            
            Neurons = new float[Layers.Count][];
            NeuronsBackprop = new float[Layers.Count][];
            NeuronDerivate = new float[Layers.Count][];
            Bias = new float[Layers.Count][];
            Weights = new float[Layers.Count][][];
            WeightCache = new float[Layers.Count][][];
            BiasCache = new float[Layers.Count][];
            Activations = new Activator[Layers.Count];

            for (int i = 0; i < Layers.Count; i++)
            {
                Neurons[i] = new float[Layers[i].Item1];
                NeuronsBackprop[i] = new float[Layers[i].Item1];
                NeuronDerivate[i] = new float[Layers[i].Item1];
                Bias[i] = Utils.Generate_Vector(Layers[i].Item1, 0.1, 0.9);
                BiasCache[i] = new float[Layers[i].Item1];
                Weights[i] = new float[Layers[i].Item1][];
                WeightCache[i] = new float[Layers[i].Item1][];
                Activations[i] = Layers[i].Item2;
            }

            for (int i = 0; i < Layers.Count; i++)
            {
                for (int j = 0; j < Layers[i].Item1; j++)
                {
                    if (i == 0)
                    {
                        Weights[i][j] = Utils.Generate_Vector(inputSize, 0.1, 0.9);
                        WeightCache[i][j] = new float[inputSize];
                    }
                    else
                    {
                        Weights[i][j] = Utils.Generate_Vector(Layers[i - 1].Item1, 0.1, 0.9);
                        WeightCache[i][j] = new float[Layers[i - 1].Item1];
                    }
                }
            }
        }

        public void Test(Dataset inputs, CostFunction costFunction)
        {
            var Accuracy = 0f;
            var TotalError = 0f;

            inputs.Shuffle(inputs.ValidationgData);

            foreach (var input in inputs.ValidationgData)
            {
                //Forward

                for (int i = 0; i < Weights[0].Length; i++)
                {
                    //Neurons[0][i] = Bias[0][i] + Dot(input.Data, Weights[0][i]);
                    Neurons[0][i] = Dot(input.Data, Weights[0][i]);
                    Neurons[0][i] = Activate(Neurons[0][i], Activations[0]);
                    NeuronDerivate[0][i] = Derivate(Neurons[0][i], Activations[0]);
                }

                for (int i = 1; i < Weights.Length; i++)
                {
                    for (int j = 0; j < Weights[i].Length; j++)
                    {
                        //Neurons[i][j] = Bias[i][j] + Dot(Neurons[i - 1], Weights[i][j]);
                        Neurons[i][j] = Dot(Neurons[i - 1], Weights[i][j]);
                        Neurons[i][j] = Activate(Neurons[i][j], Activations[i]);
                        NeuronDerivate[i][j] = Derivate(Neurons[i][j], Activations[i]);
                    }
                }

                for (int i = 0; i < Neurons[Neurons.Length - 1].Length; i++)
                {
                    Neurons[Neurons.Length - 1][i] = (float)Math.Round(Neurons[Neurons.Length - 1][i]);
                }
                var NetworkOutput = Neurons[Neurons.Length - 1];

                NetworkOutput.SequenceEqual(input.Y);

                if (NetworkOutput.SequenceEqual(input.Y))
                {
                    Accuracy++;
                }

                NeuronsBackprop[Neurons.Length - 1] = LossFunction(Neurons[Neurons.Length - 1], input.Y, costFunction);
                TotalError += NeuronsBackprop[Neurons.Length - 1].Sum();
            }
            Console.WriteLine("TEST: " + Accuracy / inputs.ValidationgData.Length);
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="learningRate"></param>
        /// <param name="epoch"></param>
        /// <param name="costFunction"></param>
        /// <param name="miniBatch"></param>
        public void Train(Dataset inputs, float learningRate, int epoch, CostFunction costFunction, int miniBatch = 32)
        {
            if (Neurons[Neurons.Length - 1].Length != inputs.TrainingData[0].Y.Length)
            {
                throw new Exception("Wrong last layer size!!! network output is " + Neurons[Neurons.Length - 1].Length + " and dataset output is " + inputs.TrainingData[0].Y.Length);
            }

            var lastIndex = Neurons.Length - 1;
            var minib = 0;
            var TotalEpochError = 0f;

            var counter = 0;

            double[,] PlotData = new double[epoch, 1];

            for (int e = 0; e < epoch; e++)
            {
                Console.WriteLine(e);
                TotalEpochError = 0;
                inputs.Shuffle(inputs.TrainingData);

                foreach (var input in inputs.TrainingData)
                 {
                //for (int inp = 0; inp < inputs.TrainingData.Length; inp++)
                //{
                    // KEKSI dropout!
                   // var input = inputs.TrainingData[inp];

                    counter++;
                    //Forward

                    for (int i = 0; i < Weights[0].Length; i++)
                    {
                        Neurons[0][i] = Bias[0][i] + Dot(input.Data, Weights[0][i]);
                        //Neurons[0][i] = Dot(input.Data, Weights[0][i]);
                        Neurons[0][i] = Activate(Neurons[0][i], Activations[0]);
                        NeuronDerivate[0][i] = Derivate(Neurons[0][i], Activations[0]);
                    }

                    for (int i = 1; i < Weights.Length; i++)
                    {
                        for (int j = 0; j < Weights[i].Length; j++)
                        {
                            Neurons[i][j] = Bias[i][j] + Dot(Neurons[i - 1], Weights[i][j]);
                            //Neurons[i][j] = Dot(Neurons[i - 1], Weights[i][j]);
                            Neurons[i][j] = Activate(Neurons[i][j], Activations[i]);
                            NeuronDerivate[i][j] = Derivate(Neurons[i][j], Activations[i]);
                        }
                    }

                    // Calculate error

                    NeuronsBackprop[Neurons.Length - 1] = LossFunction(Neurons[Neurons.Length - 1], input.Y, costFunction);
                    TotalEpochError += NeuronsBackprop[Neurons.Length - 1].Sum();

                    if (counter == 10)
                    {
                        PlotData[counter, 0] = NeuronsBackprop[Neurons.Length - 1].Sum();
                        counter = 0;
                    }

                    //Console.WriteLine(NeuronsBackprop[Neurons.Length - 1][0]);

                    //Backward

                    // layer output

                    // toimii

                    for (int i = Weights.Length - 1; i > 0; i--)
                    {
                        // Console.WriteLine("Layer: " + i);
                        for (int j = 0; j < Weights[i].Length; j++)
                        {
                            BiasCache[i][j] += Bias[i][j] * NeuronsBackprop[i][j];
                            for (int k = 0; k < Weights[i][j].Length; k++)
                            {
                                //    Console.WriteLine("WC :" + i + "," + j + "," + k + " += " + "neuron: " + i + "," + j + " * " + "NeuronBP: " + (i - 1) + "," + k);
                                //NeuronsBackprop[i - 1][k] += Weights[i][j][k] * (NeuronsBackprop[i][j] * NeuronDerivate[i][j]);
                                NeuronsBackprop[i - 1][k] += Weights[i][j][k] * NeuronsBackprop[i][j];
                                WeightCache[i][j][k] += Neurons[i][j] * NeuronsBackprop[i - 1][k];
                            }
                        }
                    }

                    /*
                    for (int i = Neurons.Length - 1; i > 0; i--)
                    {
                        Console.WriteLine("Layer: "+i);
                        for (int j = 0; j < Neurons[i].Length; j++)
                        {
                            for (int k = 0; k < Weights[i][j].Length; k++)
                            {
                                Console.WriteLine("WC :"+i+","+j+","+k+" += "+"neuron: "+i+","+j+" * "+"NeuronBP: "+(i-1)+","+k);
                                WeightCache[i][j][k] += Neurons[i][j] * NeuronsBackprop[i-1][k];
                            }
                        }
                    }
                    */
                    // Console.WriteLine("Layer: " +0);
                    for (int j = 0; j < Neurons[0].Length; j++)
                    {
                        BiasCache[0][j] += Bias[0][j] * NeuronsBackprop[0][j];

                        for (int k = 0; k < Weights[0][j].Length; k++)
                        {
                            //       Console.WriteLine("WC :" + 0 + "," + j + "," + k + " += " + "neuron: " + 0 + "," + j + " * " + "NeuronBP: " + (0 - 1) + "," + k);
                            //WeightCache[0][j][k] += input.Data[k] * (NeuronsBackprop[0][j] * NeuronDerivate[0][j]);
                            WeightCache[0][j][k] += -input.Data[k] * NeuronsBackprop[0][j];
                        }
                    }

                    minib++;

                    if (minib == miniBatch && miniBatch > 0)
                    {
                        for (int i = 0; i < Weights.Length; i++)
                        {
                            for (int j = 0; j < Weights[i].Length; j++)
                            {
                                Bias[i][j] -= ((BiasCache[i][j] / miniBatch) * learningRate);
                                BiasCache[i][j] = 0;
                                for (int k = 0; k < Weights[i][j].Length; k++)
                                {
                                    Weights[i][j][k] -= ((WeightCache[i][j][k] / miniBatch) * learningRate);
                                    WeightCache[i][j][k] = 0;
                                }
                            }
                        }
                        minib = 0;
                    }
                }

                if (miniBatch < 1)
                {
                    var lenght = inputs.TrainingData.Length;

                    for (int i = 0; i < Weights.Length; i++)
                    {
                        for (int j = 0; j < Weights[i].Length; j++)
                        {
                            BiasCache[i][j] /= lenght;
                            Bias[i][j] -= BiasCache[i][j] * learningRate;
                            BiasCache[i][j] = 0;

                            for (int k = 0; k < Weights[i][j].Length; k++)
                            {
                                WeightCache[i][j][k] /= lenght;
                                Weights[i][j][k] -= WeightCache[i][j][k] * learningRate;
                                WeightCache[i][j][k] = 0;
                            }
                        }
                    }
                }

                PlotData[e, 0] = TotalEpochError / inputs.TrainingData.Length;
                Console.WriteLine(TotalEpochError / inputs.TrainingData.Length);
                Test(inputs, costFunction);
                
                Console.ReadKey();
            }
            Plot.Graph(PlotData,learningRate,miniBatch);
        }

        private float[] Multiply(float[] a, float b)
        {
            var result = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * b;
            }

            return result;
        }

        private float[] LossFunction(float[] neuron, float[] y, CostFunction cost)
        {
            var result = new float[neuron.Length];
            switch (cost)
            {
                case CostFunction.MSE:
                    for (int i = 0; i < neuron.Length; i++)
                    {
                         result[i] = (float)(0.5 * (Math.Pow((y[i] - neuron[i]), 2)));
                        //result[i] = 0.5F * (float)Math.Pow(neuron[i] - y[i], 2);
                    }
                    return result;

                case CostFunction.MAE:
                    for (int i = 0; i < neuron.Length; i++)
                    {
                        result[i] = neuron[i] - y[i];
                    }
                    return result;

                case CostFunction.CEntropy:
                    for (int i = 0; i < neuron.Length; i++)
                    {
                        result[i] = (float)-(y[i] * Math.Log(neuron[i]));
                    }
                    return result;

                case CostFunction.Logistic:
                    for (int i = 0; i < neuron.Length; i++)
                    {
                        result[i] = (float)-Math.Log10(1 - neuron[i]);
                    }
                    return result;

                default:
                    return result;
            }
        }

        private float Derivate(float neuron, Activator activator)
        {
            switch (activator)
            {
                case Activator.Sigmoid:
                    return (float)(1 / (1 + Math.Exp(-neuron)) * (1 - (1 / (1 + Math.Exp(-neuron)))));

                case Activator.Relu:

                    if (neuron < 0)
                    {
                        neuron = 0;
                    }

                    return neuron;

                default:
                    return neuron;
            }
        }

        private float Activate(float neuron, Activator activator)
        {
            switch (activator)
            {
                case Activator.Sigmoid:
                    return (float)(1 / (1 + Math.Exp(-neuron)));

                case Activator.Relu:
                    if (neuron > 0)
                    {
                        return neuron = 1;
                    }
                    else
                    {
                        return neuron = 0;
                    }
                default:
                    return neuron;
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

            /*
            var temp = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                for (int j = 0; j < b.Length; j++)
                {
                    temp += a[i] * b[j];
                }
            }
            return temp;
            */
        }
    }
}
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace vNet
{
    internal enum Activation
    {
        None,
        Sigmoid,
        Relu,
        Swish
    }

    internal class Layer
    {
        // public Neuron[] Neurons { get; private set; }

        public float[][] Weights { get; private set; }
        public float[][] WeightsCache { get; private set; }


        public float[] Bias { get; private set; }
        public float[] BiasCache { get; private set; }

        public float[] Output { get; private set; }

        public float[] Derivate { get; private set; }

        public float[] ActivationDerivate { get; private set; }

        public int NumOfNeurons { get; private set; }

        private Activation Activation;

        public Layer(int numOfNeurons, Activation activator)
        {
            NumOfNeurons = numOfNeurons;
            Activation = activator;
        }

        public void InitLayer(int inputCount)
        {
            //Neurons = new Neuron[NumOfNeurons];
            Weights = new float[NumOfNeurons][];
            WeightsCache = new float[NumOfNeurons][];

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Utils.Generate_Vector(inputCount, 0.001, 0.009);
                
                WeightsCache[i] = new float[inputCount];

            }

            Bias = Utils.Generate_Vector(NumOfNeurons);
            BiasCache = new float[NumOfNeurons];

            Output = new float[NumOfNeurons];
            ActivationDerivate = new float[NumOfNeurons];
            Derivate = new float[inputCount];

            //for (int i = 0; i < Neurons.Length; i++) { Neurons[i] = new Neuron(inputCount); }
        }

        private float[] ActivateOutput(float[] n)
        {
            switch (Activation)
            {
                case Activation.Sigmoid:

                    for (int i = 0; i < n.Length; i++)
                    {
                        n[i] = (float)(1 / (1 + Math.Exp(-n[i])));
                    }

                    return n;

                case Activation.Swish:

                    for (int i = 0; i < n.Length; i++)
                    {
                        n[i] = (float)(n[i] * (1 / (1 + Math.Exp(-n[i]))));
                    }

                    return n;

                case Activation.Relu:

                    for (int i = 0; i < n.Length; i++)
                    {

                        if(n[i] > 0)
                        {
                            n[i] = 1;
                        }
                        else
                        {
                            n[i] = 0;
                        }
                        
                    }

                    return n;

                default:
                    return n;
            }
        }

        private float[] DerivateOutput(float[] n)
        {
            switch (Activation)
            {
                case Activation.Sigmoid:

                    for (int i = 0; i < n.Length; i++)
                    {
                        n[i] = (float)(1 / (1 + Math.Exp(-n[i])) * (1 - (1 / (1 + Math.Exp(-n[i])))));
                    }

                    return n;

                case Activation.Swish:

                    for (int i = 0; i < n.Length; i++)
                    {
                        var swish = n[i] * (1 + Math.Exp(-n[i]));
                        var sigmoid = 1 / (1 + Math.Exp(-n[i]));
                        var result = swish + (sigmoid * (1 - swish));
                        n[i] = (float)result;
                    }
                    return n;

                case Activation.Relu:

                    for (int i = 0; i < n.Length; i++)
                    {

                        if (n[i] < 0)
                        {
                            n[i] = 0;
                        }
                    }

                    return n;

                default:
                    return n;
            }
        }

        public void forward(float[] Input)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                for (int j = 0; j < Weights[i].Length; j++)
                {
                    Output[i] += Weights[i][j] * Input[j];
                }
                Output[i] += Bias[i];
                Output = ActivateOutput(Output);
            }
        }

        public void backward(float[] nextLayerOutput, float[] derivError)
        {
            // ActivationDerivate = DerivateOutput(Output);

            lock (WeightsCache)
            {
                for (int i = 0; i < WeightsCache.Length; i++)
                {

                    for (int j = 0; j < WeightsCache[i].Length; j++)
                    {
                        //Console.WriteLine(i+"-"+j);
                        /*
                        WeightsCache[i][j] += nextLayerOutput[j] * (derivError[i] * ActivationDerivate[i]);
                        BiasCache[i] += Bias[i] * (derivError[i] * ActivationDerivate[i]);
                        Derivate[i] += Weights[i][j] * derivError[i];
                        */

                        WeightsCache[i][j] += nextLayerOutput[j] * derivError[i];
                        BiasCache[i] += Bias[i] * derivError[i];
                        Derivate[i] += Weights[i][j] * derivError[i];

                    }

                }
            }
                 

        }

        public void updateWeights(float lr)
        {
            lock (Weights)
            {
                for (int i = 0; i < Weights.Length; i++)
                {
                    Bias[i] -= BiasCache[i] * lr;
                    BiasCache[i] = 0;
                    for (int j = 0; j < Weights[i].Length; j++)
                    {
                        Weights[i][j] -= WeightsCache[i][j] * lr;
                        WeightsCache[i][j] = 0;
                    }
                }
            }
        }
    }
}
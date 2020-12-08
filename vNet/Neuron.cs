using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using vNet.Activations;

namespace vNet
{
    public class Neuron
    {
        //publics
        public double Derivate, A, Z;

        public int[] ConnectionPattern;
        public double[] Derivates, Weights;

        //privates
        private double Bias, BiasCache, PrevUpdateBias;

        private readonly double[] WeightCache;
        private readonly double[] PrevUpdateRate;
        private bool DeltaSet;

        public Neuron(int connections)
        {
            Z = 0;
            A = 0;
            Bias = 0.5f;
            Weights = Utils.Generate_Vector(connections);
            WeightCache = new double[connections];
            Derivates = new double[connections];
            BiasCache = 0;
            PrevUpdateRate = new double[connections];
            PrevUpdateBias = 0;
            DeltaSet = false;
            ConnectionPattern = null;
        }

        public Neuron(int[] connectionPattern)
        {
            Z = 0;
            A = 0;
            Bias = 0.5f;
            Weights = Utils.Generate_Vector(connectionPattern.Length);
            WeightCache = new double[connectionPattern.Length];
            Derivates = new double[connectionPattern.Length];
            BiasCache = 0;
            PrevUpdateRate = new double[connectionPattern.Length];
            PrevUpdateBias = 0;
            DeltaSet = false;
            ConnectionPattern = connectionPattern;
        }

        public void ForwardCalculation(double[] input)
        {
            if (Vector.IsHardwareAccelerated)
            {
                Z = 0;
                var offset = Vector<double>.Count;

                int i = 0;
                for (i = 0; i + offset < input.Length; i += offset)
                {
                    var v1 = new Vector<double>(input, i);
                    var v2 = new Vector<double>(Weights, i);

                    Z += Vector.Dot(v1, v2);
                }
                for (; i < input.Length; ++i)
                {
                    Z += input[i] * Weights[i];
                }

                Z += Bias;
            }
            else
            {
                Z = 0;

                for (int i = 0; i < input.Length; i++)
                {
                    Z += input[i] * Weights[i];
                }

                Z += Bias;
            }
        }

        public void Backpropagate(double[] inputToNeuron)
        {
            BiasCache += Bias * Derivate;

            var offset = Vector<double>.Count;
            int i = 0;
            for (i = 0; i + offset < inputToNeuron.Length; i += offset)
            {
                var input = new Vector<double>(inputToNeuron, i);
                var res = new Vector<double>(WeightCache, i);
                Vector.Add(res, Vector.Multiply(input, Derivate)).CopyTo(WeightCache, i);
            }

            //remaining items
            for (; i < inputToNeuron.Length; ++i)
            {
                WeightCache[i] += inputToNeuron[i] * Derivate;
            }

            /*
            for (i = 0; i < WeightCache.Length; i++)
            {
                //WeightCache[i] += inputToNeuron[i] * Derivate;
                Derivates[i] = Weights[i] * Derivate;
            }
            */
        }

        public void AdjustWeights(int mbatch, double learningrate, double momentum, bool L2)
        {
            var lambda = 0d;

            // L2 regularization condition
            if (L2)
            {
                for (int i = 0; i < Weights.Length; i++)
                {
                    lambda += (Weights[i] * Weights[i]);
                }
                lambda = 1 - ((lambda * learningrate) / Weights.Length);
            }
            else
            {
                // if not L2, then multiply weights with 1
                lambda = 1;
            }

            if (!DeltaSet)
            {
                for (int i = 0; i < Weights.Length; i++)
                {
                    PrevUpdateRate[i] = (WeightCache[i] / mbatch) * learningrate;
                    Weights[i] = (Weights[i] * lambda) - PrevUpdateRate[i]; /// L2
                    WeightCache[i] = 0;
                }

                PrevUpdateBias = (BiasCache / mbatch) * learningrate;
                Bias -= PrevUpdateBias;
                BiasCache = 0;

                DeltaSet = true;
            }
            else
            {
                for (int i = 0; i < Weights.Length; i++)
                {
                    var Mom = PrevUpdateRate[i] * momentum;
                    PrevUpdateRate[i] = (WeightCache[i] / mbatch) * learningrate;
                    Weights[i] = (Weights[i] * lambda) - (PrevUpdateRate[i] + Mom);

                    //Weights[i] = Weights[i] - (WeightCache[i] / mbatch) * learningrate;
                    WeightCache[i] = 0;
                }

                var BiasMomentum = PrevUpdateBias * momentum;
                PrevUpdateBias = (BiasCache / mbatch) * learningrate;
                Bias -= PrevUpdateBias + BiasMomentum;
                BiasCache = 0;
            }
        }
    }
}
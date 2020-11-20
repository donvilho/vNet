using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using vNet.Activations;

namespace vNet
{
    internal class Neuron
    {
        public float Derivate;
        public float A;
        public float Z;

        private float Bias;
        private float BiasCache;

        public float[] Derivates;

        private float[] Weights;
        private float[] WeightCache;

        private float[] PrevUpdateRate;
        private float PrevUpdateBias;

        private bool DeltaSet;

        public int[] ConnectionPattern;

        public Neuron(int connections, bool constInit, float initVal)
        {
            Z = 0;
            A = 0;
            Bias = 1;
            Weights = Utils.Generate_Vector(connections, setNumber: constInit, number: initVal);
            WeightCache = new float[connections];
            Derivates = new float[connections];
            BiasCache = 0;
            PrevUpdateRate = new float[connections];
            PrevUpdateBias = 0;
            DeltaSet = false;
            ConnectionPattern = null;
        }

        public Neuron(int[] connectionPattern, bool constInit, float initVal)
        {
            Z = 0;
            A = 0;
            Bias = 1;
            Weights = Utils.Generate_Vector(connectionPattern.Length, setNumber: constInit, number: initVal);
            WeightCache = new float[connectionPattern.Length];
            Derivates = new float[connectionPattern.Length];
            BiasCache = 0;
            PrevUpdateRate = new float[connectionPattern.Length];
            PrevUpdateBias = 0;
            DeltaSet = false;
            ConnectionPattern = connectionPattern;
        }

        public void ForwardCalculation(float[] input)
        {
            Z = 0f;
            Z += Bias;

            if (ConnectionPattern != null)
            {
                for (int i = 0; i < ConnectionPattern.Length; i++)
                {
                    Z += (input[ConnectionPattern[i]] * Weights[i]);
                }
            }
            else
            {
                for (int i = 0; i < Weights.Length; i++)
                {
                    Z += (input[i] * Weights[i]);
                }
            }
        }

        public void Backpropagate(float[] inputToNeuron)
        {
            BiasCache += Bias * Derivate;

            var partitioner = Partitioner.Create(0, WeightCache.Length);
            /*
            Parallel.ForEach(partitioner, range =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                {
                    WeightCache[i] += (inputToNeuron[i] * Derivate);
                    Derivates[i] = Weights[i] * Derivate;
                }
            });
            */

            for (int i = 0; i < WeightCache.Length; i++)
            {
                WeightCache[i] += (inputToNeuron[i] * Derivate);
                Derivates[i] = Weights[i] * Derivate;
            }
        }

        public void AdjustWeights(int mbatch, float learningrate, float momentum)
        {
            var len = Weights.Length;

            if (!DeltaSet)
            {
                for (int i = 0; i < len; i++)
                {
                    PrevUpdateRate[i] = (WeightCache[i] / mbatch) * learningrate;
                    Weights[i] -= PrevUpdateRate[i];
                    WeightCache[i] = 0;
                }
                PrevUpdateBias = (BiasCache / mbatch) * learningrate;
                Bias -= PrevUpdateBias;
                BiasCache = 0;

                DeltaSet = true;
            }
            else
            {
                for (int i = 0; i < len; i++)
                {
                    var Mom = PrevUpdateRate[i] * momentum;
                    PrevUpdateRate[i] = (WeightCache[i] / mbatch) * learningrate;
                    Weights[i] -= PrevUpdateRate[i] + Mom;
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
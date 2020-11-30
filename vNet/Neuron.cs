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
        public float Derivate, A, Z;

        public int[] ConnectionPattern;
        public float[] Derivates, Weights;

        //privates
        private float Bias, BiasCache, PrevUpdateBias;

        private readonly float[] WeightCache;
        private readonly float[] PrevUpdateRate;
        private bool DeltaSet;

        public Neuron(int connections)
        {
            Z = 0;
            A = 0;
            Bias = 0.5f;
            Weights = Utils.Generate_Vector(connections);
            WeightCache = new float[connections];
            Derivates = new float[connections];
            BiasCache = 0;
            PrevUpdateRate = new float[connections];
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
            if (Vector.IsHardwareAccelerated)
            {
                Z = 0;
                var offset = Vector<float>.Count;

                int i = 0;
                for (i = 0; i + offset < input.Length; i += offset)
                {
                    var v1 = new Vector<float>(input, i);
                    var v2 = new Vector<float>(Weights, i);

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

        public void Backpropagate(float[] inputToNeuron)
        {
            BiasCache += Bias * Derivate;
            //WeightCache = SimdVectorAdd(WeightCache, SimdVectorScalar(inputToNeuron, Derivate));
            //Derivates = SimdVectorScalar(Weights, Derivate);

            for (int i = 0; i < WeightCache.Length; i++)
            {
                WeightCache[i] += inputToNeuron[i] * Derivate;
                Derivates[i] = Weights[i] * Derivate;
            }
        }

        public void AdjustWeights(int mbatch, float learningrate, float momentum, bool L2)
        {
            var len = Weights.Length;

            var lambda = 0f;

            // L2 regularization condition
            if (L2)
            {
                for (int i = 0; i < len; i++)
                {
                    lambda += (Weights[i] * Weights[i]);
                }
                lambda = 1 - ((lambda * learningrate) / len);
            }
            else
            {
                // if not L2, then multiply weights with 1
                lambda = 1;
            }

            if (!DeltaSet)
            {
                /*
                PrevUpdateRate = SimdVectorScalar(SimdVectorScalar(WeightCache, (1f / mbatch)), learningrate);
                Weights = SimdVectorSub(Weights, PrevUpdateRate);
                Array.Clear(WeightCache, 0, WeightCache.Length);
                */

                for (int i = 0; i < len; i++)
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
                /*
                var Mom = SimdVectorScalar(PrevUpdateRate, momentum);
                PrevUpdateRate = SimdVectorScalar(SimdVectorScalar(WeightCache, (1f / mbatch)), learningrate);
                Weights = SimdVectorSub(Weights, PrevUpdateRate);
                Weights = SimdVectorAdd(Weights, Mom);
                Array.Clear(WeightCache, 0, WeightCache.Length);
                */

                for (int i = 0; i < len; i++)
                {
                    var Mom = PrevUpdateRate[i] * momentum;
                    PrevUpdateRate[i] = (WeightCache[i] / mbatch) * learningrate;
                    Weights[i] = (Weights[i] * lambda) - (PrevUpdateRate[i] + Mom);
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
using System;
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

        public Neuron(int connections)
        {
            Z = 0;
            A = 0;

            Bias = 1;
            Weights = Utils.Generate_Vector(connections, setNumber: true, number: 0.01f);

            WeightCache = new float[connections];
            Derivates = new float[connections];

            BiasCache = 0;

            PrevUpdateRate = new float[connections];
            PrevUpdateBias = 0;
            DeltaSet = false;
        }

        public void ForwardCalculation(float[] input)
        {
            Z = 0f;
            Z += Bias;

            if (input.Length == Weights.Length)
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

            for (int i = 0; i < WeightCache.Length; i++)
            {
                WeightCache[i] += (inputToNeuron[i] * Derivate);
                Derivates[i] = Weights[i] * Derivate;
            }
        }

        public unsafe void AdjustWeights(int mbatch, float learningrate)
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
                    var momentum = PrevUpdateRate[i] * 0.5f;
                    PrevUpdateRate[i] = (WeightCache[i] / mbatch) * learningrate;
                    Weights[i] -= PrevUpdateRate[i] + momentum;
                    WeightCache[i] = 0;
                }
                var BiasMomentum = PrevUpdateBias * 0.5f;
                PrevUpdateBias = (BiasCache / mbatch) * learningrate;
                Bias -= PrevUpdateBias + BiasMomentum;
                BiasCache = 0;
            }
        }
    }
}
using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using vNet.Activations;

namespace vNet
{
    internal class Neuron
    {
        public float Derivate, A, Z;
        private int Offset;

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
            //Derivates = SimdVectorScalar(Weights, Derivate);

            if (Vector.IsHardwareAccelerated)
            {
                WeightCache = SimdVectorAdd(WeightCache, SimdVectorScalar(inputToNeuron, Derivate));
            }
            else
            {
                for (int i = 0; i < WeightCache.Length; i++)
                {
                    WeightCache[i] += inputToNeuron[i] * Derivate;
                    //Derivates[i] = Weights[i] * Derivate;
                }
            }
        }

        public void AdjustWeights(int mbatch, float learningrate, float momentum)
        {
            var len = Weights.Length;

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
                    Weights[i] -= PrevUpdateRate[i] + Mom;
                    WeightCache[i] = 0;
                }

                var BiasMomentum = PrevUpdateBias * momentum;
                PrevUpdateBias = (BiasCache / mbatch) * learningrate;
                Bias -= PrevUpdateBias + BiasMomentum;
                BiasCache = 0;
            }
        }

        private float[] SimdVectorAddScalar(float[] result, float[] left, float right)
        {
            var offset = Vector<float>.Count;
            int i = 0;
            for (i = 0; i + offset < left.Length; i += offset)
            {
                var v1 = new Vector<float>(left, i);
                var res = new Vector<float>(result, i);
                Vector.Add(res, Vector.Multiply(v1, right)).CopyTo(result, i);
            }

            //remaining items
            for (; i < left.Length; ++i)
            {
                result[i] += left[i] * right;
            }

            return result;
        }

        private float[] SimdVectorScalar(float[] left, float right)
        {
            var offset = Vector<float>.Count;
            float[] result = new float[left.Length];
            int i = 0;
            for (i = 0; i + offset < left.Length; i += offset)
            {
                var v1 = new Vector<float>(left, i);
                Vector.Multiply(v1, right).CopyTo(result, i);
            }

            //remaining items
            for (; i < left.Length; ++i)
            {
                result[i] += left[i] * right;
            }

            return result;
        }

        private float[] SimdVectorAdd(float[] left, float[] right)
        {
            var offset = Vector<float>.Count;
            float[] result = new float[left.Length];
            int i = 0;
            for (i = 0; i + offset < left.Length; i += offset)
            {
                var v1 = new Vector<float>(left, i);
                var v2 = new Vector<float>(right, i);

                Vector.Add(v1, v2).CopyTo(result, i);
            }

            //remaining items
            for (; i < left.Length; ++i)
            {
                result[i] = left[i] + right[i];
            }

            return result;
        }

        private float[] SimdVectorSub(float[] left, float[] right)
        {
            var offset = Vector<float>.Count;
            float[] result = new float[left.Length];
            int i = 0;
            for (i = 0; i + offset < left.Length; i += offset)
            {
                var v1 = new Vector<float>(left, i);
                var v2 = new Vector<float>(right, i);

                Vector.Subtract(v1, v2).CopyTo(result, i);
            }

            //remaining items
            for (; i < left.Length; ++i)
            {
                result[i] = left[i] - right[i];
            }

            return result;
        }

        private float[] SimdVectorDivision(float[] left, float[] right)
        {
            var offset = Vector<float>.Count;
            float[] result = new float[left.Length];
            int i = 0;
            for (i = 0; i + offset < left.Length; i += offset)
            {
                var v1 = new Vector<float>(left, i);
                var v2 = new Vector<float>(right, i);

                Vector.Divide(v1, v2).CopyTo(result, i);
            }

            //remaining items
            for (; i < left.Length; ++i)
            {
                result[i] = left[i] / right[i];
            }

            return result;
        }

        private float SimdVectorProd(float[] left, float[] right)
        {
            var offset = Vector<float>.Count;
            float result = 0;
            int i = 0;
            for (i = 0; i + offset < left.Length; i += offset)
            {
                var v1 = new Vector<float>(left, i);
                var v2 = new Vector<float>(right, i);

                result += Vector.Dot(v1, v2);
            }

            //remaining items
            for (; i < left.Length; ++i)
            {
                result += left[i] * right[i];
            }

            return result;
        }
    }
}
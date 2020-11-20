using System;
using System.Collections.Concurrent;
using System.Numerics;
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
                /*
                for (int i = 0; i < ConnectionPattern.Length; i++)
                {
                    Z += input[ConnectionPattern[i]] * Weights[i];
                }
                */

                var offset = Vector<float>.Count;
                Z = 0f;
                int i = 0;
                for (i = 0; i < input.Length; i += offset)
                {
                    var v1 = new Vector<float>(input, i);
                    var v2 = new Vector<float>(Weights, i);

                    Z += Vector.Dot(v1, v2);
                }

                //remaining items
                for (; i < input.Length; ++i)
                {
                    Z += input[i] * Weights[i];
                }
            }
            else
            {
                SimdVectorProd(input, Weights);

                /*
                for (int i = 0; i < Weights.Length; i++)
                {
                    Z += input[i] * Weights[i];
                }
                */
            }
        }

        public void Backpropagate(float[] inputToNeuron)
        {
            BiasCache += Bias * Derivate;
            Derivates = SimdVectorScalar(Weights, Derivate);

            // WeightCache = SimdVectorAddScalar(WeightCache, Weights, Derivate);
            /*
            for (int i = 0; i < WeightCache.Length; i++)
            {
                WeightCache[i] += inputToNeuron[i] * Derivate;
                // Derivates[i] = Weights[i] * Derivate;
            }
            */

            var offset = Vector<float>.Count;
            int i = 0;
            for (i = 0; i < inputToNeuron.Length; i += offset)
            {
                var v1 = new Vector<float>(inputToNeuron, i);
                var res = new Vector<float>(WeightCache, i);
                Vector.Add(res, Vector.Multiply(v1, Derivate)).CopyTo(WeightCache, i);
            }

            //remaining items
            for (; i < inputToNeuron.Length; ++i)
            {
                WeightCache[i] += inputToNeuron[i] * Derivate;
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

        public void Dot()
        {
            /*
            int vectorSize = Vector<float>.Count;
            var accVector = Vector<float>.Zero;
            int i;
            var array = Neuron;
            for (i = 0; i <= array.Length - vectorSize; i += vectorSize)
            {
                var v = new Vector<int>(array, i);
                accVector = Vector.Add(accVector, v);
            }
            int result = Vector.Dot(accVector, Vector<int>.One);
            for (; i < array.Length; i++)
            {
                result += array[i];
            }
            return result;
            */
        }

        private float[] SimdVectorAddScalar(float[] result, float[] left, float right)
        {
            var offset = Vector<float>.Count;
            int i = 0;
            for (i = 0; i < left.Length; i += offset)
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
            for (i = 0; i < left.Length; i += offset)
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

        private void SimdVectorProd(float[] left, float[] right)
        {
            var offset = Vector<float>.Count;
            Z = 0f;
            int i = 0;
            for (i = 0; i < left.Length; i += offset)
            {
                var v1 = new Vector<float>(left, i);
                var v2 = new Vector<float>(right, i);

                Z += Vector.Dot(v1, v2);
            }

            //remaining items
            for (; i < left.Length; ++i)
            {
                Z += left[i] * right[i];
            }

            //return result;
        }
    }
}
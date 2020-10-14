using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    class Neuron
    {
        private float Bias;
        private float BiasCache;
        private float[] Weights;
        private float[] WeightCache;


        public Neuron(int connections)
        {
            Bias = (float)new Random().NextDouble();
            Weights = Utils.Generate_Vector(connections);
            WeightCache = new float[connections];
            BiasCache = 0;
        }

        public float Activate(float[] img)
        {
            float Output = Bias;

            if(img.Length == Weights.Length)
            {
                for(int i = 0; i < Weights.Length; i++)
                {
                    Output += img[i] * Weights[i];
                }
            }
            return Output;
        }

        public float Backpropagate(float[] input, float error)
        {
            float Output = Bias;

            BiasCache += Bias * error;

            for (int i = 0; i < WeightCache.Length; i++)
            {
                WeightCache[i] += input[i] * error;
            }

            return Output;
        }

        public void AdjustWeights(int mbatch)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = WeightCache[i] / mbatch;
            }
            Bias = BiasCache / mbatch;
        }
    }
}

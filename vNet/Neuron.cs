using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{

    class Neuron
    {
        public float Derivate;
        public float Value;
        private float Bias;
        private float BiasCache;
        private float[] Weights;
        private float[] WeightCache;

        private float[] PrevUpdateRate;
        private float PrevUpdateBias;

        private bool DeltaSet;

        public Neuron(int connections)
        {
            Bias = (float)new Random().NextDouble();
            Weights = Utils.Generate_Vector(connections,0.01,0.09);
            WeightCache = new float[connections];
            BiasCache = 0;
            Value = 0;
            PrevUpdateRate = new float[connections];
            PrevUpdateBias = 0;
            DeltaSet = false;
        }


        public void ForwardCalculation(float[] input)
        {
            Value = 0f;
            //Value += Bias;
            
            if(input.Length == Weights.Length)
            {
                for(int i = 0; i < Weights.Length; i++)
                {
                    Value += (input[i] * Weights[i]);
                }
            }
        }



        public void Backpropagate(float[] inputToNeuron)
        {
            BiasCache += Bias * Derivate;

            for (int i = 0; i < WeightCache.Length; i++)
            {
                WeightCache[i] += (inputToNeuron[i] * Derivate);
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
                            var momentum = PrevUpdateRate[i]*0.5f;
                            PrevUpdateRate[i] = (WeightCache[i] / mbatch) * learningrate;
                            Weights[i] -= PrevUpdateRate[i]  ;
                            WeightCache[i] = 0;
                        }
                        var BiasMomentum = PrevUpdateBias * 0.5f;
                        PrevUpdateBias = (BiasCache / mbatch) * learningrate;
                        Bias -= PrevUpdateBias;
                        BiasCache = 0;
                    
            }
         
        }
    }
}

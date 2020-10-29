using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{

    class Network
    {
        public readonly float[] Neurons,Error,Derivate;
        public readonly float[][] Weights, WeightCache;
        public readonly float[] Bias, BiasCache;
        

        private readonly Random rnd;

        public Network(int neuronCount, int inputLenght)
        {
            rnd = new Random();
            Neurons = new float[neuronCount];
            Error = new float[neuronCount];
            Derivate = new float[neuronCount];

            Bias = Utils.Generate_Vector(neuronCount);
            BiasCache = new float[neuronCount];

            Weights = new float[neuronCount][];
            WeightCache = new float[neuronCount][];

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Utils.Generate_Vector(inputLenght, 0.0001, 0.0009);
                WeightCache[i] = new float[inputLenght];
            }
        }
        public void Train((float[], float[], string) input)
        {

            
            var ExpSum = 0f;
            var Loss = 0f;

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = Bias[i] + Utils.Dot(Weights[i], input.Item1);
                //Calc EXP SUM
                ExpSum += (float)Math.Exp(Neurons[i]);
            }

           

            for (int i = 0; i < Derivate.Length; i++)
            {
                //CalcError/activate

                Error[i] = (float)Math.Exp(Neurons[i]) / ExpSum;
                Loss += input.Item2[i] * (float)Math.Log(Error[i]);

                //Loss += input.Item2[i] * (float)Math.Log(Math.Exp(Neurons[i]) / ExpSum);
                //CalcDerivates
                //D-A
                Derivate[i] = Error[i] - input.Item2[i];
                //D-Z
                Derivate[i] *= Error[i] * (1 - Error[i]);



                for (int j = 0; j < WeightCache[i].Length; j++)
                {
                    //D-W
                    WeightCache[i][j] += input.Item1[j] * Derivate[i];
                    //D-B
                    BiasCache[i] += Bias[i] * Derivate[i];
                }
            }
            //return (WeightCache,BiasCache, Loss);
        }

        public void Update(Network[] networks, float learningRate)
        {

            for (int i = 0; i < networks.Length; i++)
            {

            }



        }
    }
}

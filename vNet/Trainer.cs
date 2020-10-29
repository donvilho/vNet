using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    
    class Trainer
    {

        private Network Network;

        public Trainer(Network net)
        {
            Network = net;
        }

        public void Train((float[], float[], string) input)
        {
            var ExpSum = 0f;
            var Loss = 0f;
 
            for (int i = 0; i < Network.Neurons.Length; i++)
            {
                Network.Neurons[i] = Network.Bias[i] + Utils.Dot(Network.Weights[i], input.Item1);
                //Calc EXP SUM
                ExpSum += (float)Math.Exp(Network.Neurons[i]);
            }

            Network.Neurons.Sum();

            for (int i = 0; i < Network.Derivate.Length; i++)
            {
                //CalcError/activate

                Network.Error[i] = (float)Math.Exp(Network.Neurons[i]) / ExpSum;
                Loss += input.Item2[i] * (float)Math.Log(Network.Error[i]);

                //Loss += input.Item2[i] * (float)Math.Log(Math.Exp(Neurons[i]) / ExpSum);
                //CalcDerivates
                //D-A
                Network.Derivate[i] = Network.Error[i] - input.Item2[i];
                //D-Z
                Network.Derivate[i] *= Network.Error[i] * (1 - Network.Error[i]);



                for (int j = 0; j < Network.WeightCache[i].Length; j++)
                {
                    //D-W
                    Network.WeightCache[i][j] += input.Item1[j] * Network.Derivate[i];
                    //D-B
                    Network.BiasCache[i] += Network.Bias[i] * Network.Derivate[i];
                }
            }
            //return (WeightCache,BiasCache, Loss);
        }

    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace vNet
{
    
    class Trainer
    {

        public readonly Network Net;
    

        public Trainer(Network net)
        {
            Net = net;
     
        }


        public void Train(Input Data)
        {

            for (int i = 0; i < Net.Neurons.Length; i++)
            {
                Net.Neurons[i].ForwardCalculation(Data.Data);
                Net.Error[i] = (float)Utils.exp4(Net.Neurons[i].Value);
            }

            var ExpSum = Net.Error.Sum();

            for (int i = 0; i < Net.Error.Length; i++)
            {
                Net.Error[i] /= ExpSum;
            }

          
            for (int i = 0; i < Net.Neurons.Length; i++)
            {
                //* (Net.Error[i] * (1 - Net.Error[i])
                Net.Neurons[i].Derivate = Net.Error[i] - Data.TruthLabel[i] ;
                //Net.Neurons[i].Derivate = -input.Item2[i] * (float)Math.Log(Net.Error[i]);
                Net.Neurons[i].Backpropagate(Data.Data);
            }

        }

        public (float, bool, int) Test(Input Data)
        {
            var Loss = 0f;

            for (int i = 0; i < Net.Neurons.Length; i++)
            {
                Net.Neurons[i].ForwardCalculation(Data.Data);
                Net.Error[i] = (float)Math.Exp(Net.Neurons[i].Value);
            }

            var ExpSum = Net.Error.Sum();

            for (int i = 0; i < Net.Error.Length; i++)
            {
                Net.Error[i] /= ExpSum;

                Loss += Data.TruthLabel[i] * (float)Math.Log(Net.Error[i]);
                
            }

            int position = 0;

            for (int i = 0; i < Net.Error.Length; i++)
            {
                if(Net.Error[i] < Net.Error.Max())
                {
                    Net.Error[i] = 0;
                }
                else
                {
                    Net.Error[i] = 1;
                    position = i;
                }
            }

            return (-Loss, Net.Error.SequenceEqual(Data.TruthLabel),position);
        }
    }
}

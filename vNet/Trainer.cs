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


        public void Train((float[], float[], string) input)
        {
            //var Loss = 0f;

            for (int i = 0; i < Net.Neurons.Length; i++)
            {
                Net.Neurons[i].ForwardCalculation(input.Item1);
                Net.Error[i] = (float)Math.Exp(Net.Neurons[i].Value);
            }

            var ExpSum = Net.Error.Sum();

            for (int i = 0; i < Net.Error.Length; i++)
            {
                Net.Error[i] /= ExpSum;
              //  Loss += input.Item2[i] * (float)Math.Log(Error[i]);
            }

            //Loss = -Loss;
           

            for (int i = 0; i < Net.Neurons.Length; i++)
            {
                //* (Net.Error[i] * (1 - Net.Error[i])
                Net.Neurons[i].Derivate = Net.Error[i] - input.Item2[i] ;
                //Net.Neurons[i].Derivate = -input.Item2[i] * (float)Math.Log(Net.Error[i]);
                Net.Neurons[i].Backpropagate(input.Item1);
            }

        }

        public (float, bool, int) Test((float[], float[], string) input)
        {
            var Loss = 0f;

            for (int i = 0; i < Net.Neurons.Length; i++)
            {
                Net.Neurons[i].ForwardCalculation(input.Item1);
                if (float.IsNaN(Net.Neurons[i].Value))
                {
                    Console.WriteLine();
                }
                Net.Error[i] = (float)Math.Exp(Net.Neurons[i].Value);
                if (float.IsNaN(Net.Error[i]))
                {
                    Console.WriteLine();
                }
            }

            var ExpSum = Net.Error.Sum();

            for (int i = 0; i < Net.Error.Length; i++)
            {
                Net.Error[i] /= ExpSum;

                
                if (float.IsNaN(input.Item2[i] * (float)Math.Log(Net.Error[i])))
                {
                    Console.WriteLine("NAN : "+input.Item2[i]+" - "+ Net.Error[i]+" threa: "+Thread.CurrentThread.ManagedThreadId);
                }
                Loss += input.Item2[i] * (float)Math.Log(Net.Error[i]);
                
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

            return (-Loss, Net.Error.SequenceEqual(input.Item2),position);
        }
    }
}

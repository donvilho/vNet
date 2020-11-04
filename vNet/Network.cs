using System;
using System.Collections.Generic;
using System.Linq;
using System.Resources;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{

    class Network
    {
        public readonly Neuron[] Neurons;
        public float[] Error;
      

        public Network(int neuronCount, int inputLenght)
        {
            Neurons = new Neuron[neuronCount];
            Error = new float[neuronCount];

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(inputLenght);
            }
        }

        public void UpdateWeights(int miniBatch, float learningrate)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].AdjustWeights(miniBatch, learningrate);
            }
        }
    }
}

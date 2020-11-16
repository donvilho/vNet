using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using vNet.Activations;
using vNet.LossFunctions;

namespace vNet
{
    internal class Layer
    {
        public readonly Neuron[] Neurons;
        public readonly Activation Activation;
        public float[] LayerOutput;

        public Layer(int neuronCount, int inputLenght, Activation act)
        {
            Activation = act;
            Neurons = new Neuron[neuronCount];

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(inputLenght);
            }
        }

        public void ForwardPropagateLayer(float[] inputData)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].ForwardCalculation(inputData);
            }

            LayerOutput = Activation.Activate(Neurons);
        }

        public void BackpropagateLayer(float[] Output)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Derivate = Activation.Derivate(Neurons[i].A, Output[i]);
            }

            for (int i = 0; i < Neurons.Length; i++)
            {
                //  Neurons[i].Backpropagate(InputCache);
            }
        }

        public void BackpropagateLayer(Layer PrevLayer)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                //   Neurons[i].Derivate = Activation.Derivate(Neurons[i].A, PrevLayerDerviOutput[i]);
            }

            for (int i = 0; i < Neurons.Length; i++)
            {
                // Neurons[i].Backpropagate(InputCache);
            }
        }
    }
}
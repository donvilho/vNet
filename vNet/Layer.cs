using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    enum Activation
    {
        None,
        Sigmoid,
    }

    class Layer
    {
        public Neuron[] Neurons { get; private set; }
        public float[] LayerOutput { get; private set; }
        public float[] LayerBackProp { get; private set; }
        public int NumOfNeurons { get; private set; }
        public Activation Activation { get; private set; }

        public Layer(int numOfNeurons, Activation activation) => (NumOfNeurons, Activation) = (numOfNeurons, activation);
    
           
        public void InitLayer(int inputCount)
        {
            Neurons = new Neuron[NumOfNeurons];
            LayerOutput = new float[NumOfNeurons];
            LayerBackProp = new float[NumOfNeurons];

            for (int i = 0; i < Neurons.Length; i++) { Neurons[i] = new Neuron(inputCount); }
        }

        public void ActivateLayer(float[] InputToLayer)
        {
            for(int i = 0; i < Neurons.Length; i++)
            { 
                LayerOutput[i] = Neurons[i].Activate(InputToLayer);
            }
        }

        public void Backpropagate(float[] InputToLayer, float[] PrevLayerError)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                LayerBackProp[i] = Neurons[i].Backpropagate(InputToLayer, PrevLayerError[i]);
            }
        }
    }
}

namespace vNet
{
    internal class Network
    {
        public readonly Layer[] Layers;
        public readonly Loss LossFunction;
        public readonly int NeuronCount;

        public Network(int neuronCount, int inputLenght, Activation activation, Loss loss)
        {
            NeuronCount = neuronCount;
            LossFunction = loss;
        }

        public Network(Layer[] layers, int inputLenght, Activation activation, Loss loss)
        {
            Layers = layers;
            LossFunction = loss;
        }

        public void TrainSample(Input sample)
        {
            var lastIndex = Layers.Length;

            for (int i = 0; i < lastIndex; i++)
            {
                if (i == 0)
                {
                    Layers[i].ForwardPropagateLayer(sample.Data);
                }
                else
                {
                    Layers[i].ForwardPropagateLayer(Layers[i - 1].LayerOutput);
                }
            }

            var LossValue = LossFunction.Calculate(Layers[lastIndex].LayerOutput, sample.TruthLabel);

            for (int i = lastIndex; i > 0; i--)
            {
                if (i == lastIndex)
                {
                    Layers[i].BackpropagateLayer(sample.TruthLabel);
                }
                else
                {
                }
            }
        }

        public void UpdateWeights(int miniBatch, float learningrate)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    //  Layers[i].Neurons[j].AdjustWeights(miniBatch, learningrate);
                }
            }
        }

        public void MakePrediction(int index)
        {
        }
    }
}
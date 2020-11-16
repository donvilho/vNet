using System;
using System.Linq;
using vNet.Activations;

namespace vNet
{
    internal class Trainer
    {
        public readonly Network Net;
        public float[] Error;

        public Trainer(Network net)
        {
            Net = net;
            Error = new float[Net.NeuronCount];
        }

        public void Train(Input Data, Activation activation)
        {
            /*
            for (int i = 0; i < Net.Neurons.Length; i++)
            {
                Net.Neurons[i].ForwardCalculation(Data.Data);
                Net.Error[i] = (float)Utils.exp6(Net.Neurons[i].Z);
            }

            // Activate Neurons

            Net.Error = activation.Activate(Net.Error);

            var Prediction = new int[Data.TruthLabel.Length];
            Prediction[Net.Error.ToList().IndexOf(Net.Error.Max())] = 1;

            for (int i = 0; i < Net.Neurons.Length; i++)
            {
                //HOX HOX TÄMÄ KOHTA!!!! Net.Error vs Prediction
                Net.Neurons[i].Derivate = activation.Derivate(Net.Error[i], Data.TruthLabel[i]);
                Net.Neurons[i].Backpropagate(Data.Data);
            }
            */
        }

        //public (float, bool, int) Test(Input Data, Activation activation)
        //{
        /*
        var loss = 0d;

        for (int i = 0; i < Net.Neurons.Length; i++)
        {
            Net.Neurons[i].ForwardCalculation(Data.Data);
            Net.Error[i] = (float)Utils.exp6(Net.Neurons[i].Z);
        }

        Net.Error = activation.Activate(Net.Error);

        // index of highest value
        var Prediction = new int[Data.TruthLabel.Length];
        int position = Net.Error.ToList().IndexOf(Net.Error.Max());
        Prediction[position] = 1;

        //loss = Net.

        var truthPosition = Data.TruthLabel.ToList().IndexOf(Data.TruthLabel.Max());

        var Match = position == truthPosition ? true : false;

        return ((float)loss, Match, position);
        */
        //}
    }
}
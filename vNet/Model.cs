using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    enum Type
    {
        Log_Reg,
        Lin_Reg
    }
    enum CostFunction
    {
        Msqrt,
    }

    class Model
    {

        public Layer[] Layers { get; private set; }
        public Dataset Dataset { get; private set; }

        public int ValidSetSize { get; private set; }
        public int MBatchSize { get; private set; }
        public CostFunction CostFunction { get; private set; }

        /// <summary>
        /// Linear Regression model
        ///
        /// </summary>
        /// <param name="dataPath">Path to dataset</param>
        /// <param name="epoch">Epoch N</param>
        /// <param name="learningRate">Learningrate</param>
        /// <param name="validationSetSize">Validationset size N</param>
        /// <param name="miniBathSize">Mini batch size n</param>
        public Model(Type modelType, CostFunction costFunction, string dataPath, int validationSetSize = 10, int miniBathSize = 32, Layer[] layers = null)
        {
            Dataset = Utils.DatasetCreator(dataPath);
            ValidSetSize = validationSetSize;
            MBatchSize = miniBathSize;
            CostFunction = costFunction;

            switch (modelType)
            {
                case Type.Lin_Reg:
                    Layers = new Layer[] { new Layer(1, Activation.Sigmoid) };
                    Layers[0].InitLayer(Dataset.InputLenght);
                    break;
                default:
                    Layers = layers;
                    Layers[0].InitLayer(Dataset.InputLenght);
                    break;
            }
        }

        public void TrainNetwork(int epoch, double learningRate)
        {

            

            for(int e = 0; e < epoch; e++)
            {
                Dataset.Shuffle();
                float epochError = 0;
                foreach(var img in Dataset.TrainingData)
                {
                    float[] error = null;

                    //Forward 
                    for (int layer = 0; layer < Layers.Length; layer++)
                    {
                        if(layer == 0)
                        {
                            Layers[layer].ActivateLayer(img.Data);
                        }
                        else
                        {
                            Layers[layer].ActivateLayer(Layers[layer-1].LayerOutput);
                        }

                        if (layer == Layers.Length - 1)
                        {
                            error = Utils.CalculateError(Layers[layer].LayerOutput, img.Y);
                            epochError += error.Sum();

                            error = Utils.VectorScalarMultiply(
                                Utils.Sigmoid_Derivate(Layers[layer].LayerOutput),
                                (float)learningRate);
                        }
                    }

                    //Backpropagate
                    for (int layer = Layers.Length-1; layer >= 0; layer--)
                    {
                        if (layer == Layers.Length-1)
                        {
                            Layers[layer].Backpropagate(img.Data , error);
                        }
                        else
                        {
                            Layers[layer].Backpropagate(Layers[layer - 1].LayerOutput, Layers[layer].LayerOutput);
                        }
                    }
                }

                Console.WriteLine(epochError);
            }
        }
    }
}

using System;

namespace vNet
{
    internal class LinearRegression : ModelType
    {
        private float[] Weights;
        private float Bias;
        private Dataset Dataset;
        private int Epoch;
        private float LearningRate;

        public LinearRegression(Dataset dataset, int epoch, float learningrate)
        {
            Weights = new float[dataset.InputLenght];
            Bias = 1;
            Dataset = dataset;
            Epoch = epoch;
            LearningRate = learningrate;
        }

        public float LossFunction(float output, float y, float[] X)
        {
            for (int i = 0; i < X.Length; i++)
            {
            }

            return (float)(0.5 * (Math.Pow((y - output), 2)));
        }

        public void TestModel()
        {
            throw new NotImplementedException();
        }

        public void TrainModel(bool Plot = false)
        {
            for (int e = 0; e < Epoch; e++)
            {
                foreach (var input in Dataset.TrainingData)
                {
                    var Output = Bias + Utils.Dot(input.Data, Weights);
                    var Error = input.TruthLabel[0] - Output;

                    Console.WriteLine(Error);
                    Console.ReadKey();
                    Bias -= (Bias * Error) * LearningRate;
                    for (int i = 0; i < Weights.Length; i++)
                    {
                        Weights[i] -= (Weights[i] * Error) * LearningRate;
                    }
                }
                Console.ReadKey();
            }
        }
    }
}
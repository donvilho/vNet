using System;
using System.Linq;

namespace vNet
{
    internal class Dataset
    {
        public Input[] TrainingData { get; private set; }
        public Input[] ValidationgData { get; private set; }
        public int InputLenght { get; private set; }
        public int classCount { get; private set; }

        public Dataset(Input[] dataset)
        {
            Shuffle(dataset);
            ValidationgData = dataset.Take((int)((float)dataset.Length * 0.2f)).ToArray();
            TrainingData = dataset.Skip((int)((float)dataset.Length * 0.2f)).ToArray();
            InputLenght = TrainingData[0].Data.Length;
            Normalize();
        }

        public Dataset(Input[] training, Input[] test)
        {
            TrainingData = training;
            ValidationgData = test;
            InputLenght = TrainingData[0].Data.Length;
            classCount = TrainingData[0].TruthLabel.Length;
        }

        public void Reduce(int value)
        {
            Shuffle(TrainingData);
            TrainingData = TrainingData.Take((TrainingData.Length / 100) * value).ToArray();
            Shuffle(ValidationgData);
            ValidationgData = ValidationgData.Take((ValidationgData.Length / 100) * value).ToArray();
        }

        public void Shuffle(Input[] Array)
        {
            var rand = new Random();
            for (int Count = Array.Length - 1; Count > 1; Count--)
            {
                int i = rand.Next(Count + 1);
                Input value = Array[i];
                Array[i] = Array[Count];
                Array[Count] = value;
            }
        }

        private void Normalize()
        {
            var colsMean = new float[TrainingData[0].Data.Length];
            var colsMax = new float[TrainingData[0].Data.Length];

            for (int i = 0; i < TrainingData.Length; i++)
            {
                for (int j = 0; j < TrainingData[i].Data.Length; j++)
                {
                    colsMean[j] += TrainingData[i].Data[j];
                    if (colsMax[j] < TrainingData[i].Data[j])
                    {
                        colsMax[j] = TrainingData[i].Data[j];
                    }
                }
            }

            for (int i = 0; i < colsMean.Length; i++)
            {
                colsMean[i] /= TrainingData.Length;
            }

            for (int i = 0; i < TrainingData.Length; i++)
            {
                for (int j = 0; j < TrainingData[i].Data.Length; j++)
                {
                    TrainingData[i].Data[j] -= colsMean[j];
                    TrainingData[i].Data[j] /= colsMax[j];
                }
            }
        }
    }
}
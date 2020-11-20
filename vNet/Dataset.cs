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
            Normalize_Datasets();
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

        public void Normalize_Datasets()
        {
            Normalize(TrainingData);
            Normalize(ValidationgData);
        }

        private void Normalize(Input[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Data.Length; j++)
                {
                    data[i].Data[j] = data[i].Data[j] > 200 ? 1 : 0;
                }
            }
        }

        public void ApplyConnectionMask(int[] mask)
        {
            for (int i = 0; i < TrainingData.Length; i++)
            {
                var newInput = new float[mask.Length];
                for (int j = 0; j < mask.Length; j++)
                {
                    newInput[j] = TrainingData[i].Data[mask[j]];
                }
                TrainingData[i].Data = newInput;
            }

            for (int i = 0; i < ValidationgData.Length; i++)
            {
                var newInput = new float[mask.Length];
                for (int j = 0; j < mask.Length; j++)
                {
                    newInput[j] = ValidationgData[i].Data[mask[j]];
                }
                ValidationgData[i].Data = newInput;
            }
        }
    }
}
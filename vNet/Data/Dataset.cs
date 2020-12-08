using System;
using System.Linq;

namespace vNet
{
    [Serializable]
    public class Dataset
    {
        public Input[] TrainingData { get; private set; }
        public Input[] ValidationData { get; private set; }
        public Input[] DevSet { get; private set; }
        public int InputLenght { get; private set; }
        public int classCount { get; private set; }
        public int[] connectionMask;

        public Dataset(string trainingset, string testset, int reduceTo = 0)
        {
            TrainingData = Utils.DataArrayCreator(trainingset);
            ValidationData = Utils.DataArrayCreator(testset);
            InputLenght = TrainingData[0].Data.Length;
            classCount = TrainingData[0].TruthLabel.Length;
            connectionMask = null;

            //Shuffle(TrainingData);
            //DevSet = TrainingData.Take((TrainingData.Length / 100) * 10).ToArray();

            if (reduceTo > 0)
            {
                ReduceToPercentage(reduceTo);
            }
        }

        public Dataset(Input[] dataset)
        {
            Shuffle(dataset);
            ValidationData = dataset.Take((int)((double)dataset.Length * 0.2f)).ToArray();
            TrainingData = dataset.Skip((int)((double)dataset.Length * 0.2f)).ToArray();
            InputLenght = TrainingData[0].Data.Length;
            Normalize_Datasets();
        }

        public Dataset(Input[] training, Input[] test)
        {
            TrainingData = training;
            ValidationData = test;
            InputLenght = TrainingData[0].Data.Length;
            classCount = TrainingData[0].TruthLabel.Length;
            connectionMask = null;
        }

        public void ReduceToPercentage(int value)
        {
            Shuffle(TrainingData);
            TrainingData = TrainingData.Take((TrainingData.Length / 100) * value).ToArray();
            Shuffle(ValidationData);
            ValidationData = ValidationData.Take((ValidationData.Length / 100) * value).ToArray();
        }

        public Dataset Copy()
        {
            Dataset newDataset = (Dataset)this.MemberwiseClone();
            newDataset.InputLenght = this.InputLenght;
            newDataset.DevSet = this.DevSet;
            newDataset.TrainingData = this.TrainingData;
            newDataset.ValidationData = this.ValidationData;
            newDataset.classCount = this.classCount;
            newDataset.connectionMask = this.connectionMask;
            return newDataset;
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
            Normalize(ValidationData);
        }

        private void Normalize(Input[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Data.Length; j++)
                {
                    data[i].Data[j] = data[i].Data[j] > 150 ? 1 : 0;
                }
            }
        }

        public void ApplyConnectionMask(int[] mask)
        {
            for (int i = 0; i < TrainingData.Length; i++)
            {
                var newInput = new double[mask.Length];
                for (int j = 0; j < mask.Length; j++)
                {
                    newInput[j] = TrainingData[i].Data[mask[j]];
                }
                TrainingData[i].Data = newInput;
            }

            for (int i = 0; i < ValidationData.Length; i++)
            {
                var newInput = new double[mask.Length];
                for (int j = 0; j < mask.Length; j++)
                {
                    newInput[j] = ValidationData[i].Data[mask[j]];
                }
                ValidationData[i].Data = newInput;
            }

            for (int i = 0; i < DevSet.Length; i++)
            {
                var newInput = new double[mask.Length];
                for (int j = 0; j < mask.Length; j++)
                {
                    newInput[j] = DevSet[i].Data[mask[j]];
                }
                DevSet[i].Data = newInput;
            }

            connectionMask = mask;
        }
    }
}
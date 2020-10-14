using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    class Dataset
    {

        public Input[] TrainingData { get; private set; }
        public Input[] ValidationgData { get; private set; }
        public int InputLenght { get; private set; }

        public Dataset(Input[] training, Input[] test)
        {
            TrainingData = training;
            ValidationgData = test;
            InputLenght = TrainingData[0].Data.Length;
        }

        public void Shuffle()
        {
            var rand = new Random();
            for (int Count = TrainingData.Length - 1; Count > 1; Count--)
            {
                int i = rand.Next(Count + 1);
                Input value = TrainingData[i];
                TrainingData[i] = TrainingData[Count];
                TrainingData[Count] = value;
            }
        }

    }
}

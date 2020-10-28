using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    class DatasetArray
    {
        public float[][] TrainingData, Training_Truth, TestData, Test_Truth;
        public object[][] Training_Label, Test_Label;

        public DatasetArray(string path)
        {


            var re = Utils.DataArrayCreator(path);

          


        }
    }
}

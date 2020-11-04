using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
   
    struct Input
    {
        public float[] TruthLabel { get; private set; }
        public float[] Data { get; private set; }
        public string LabelName { get; private set; }


        public Input(float[] data, float[] y, string labelname)
        {
            TruthLabel = y;
            LabelName = labelname;
            Data = data;
        }

        public Input(float[]data , float[] y)
        {
            TruthLabel = y;
            Data = data;
            LabelName = null;
        }

        public Input(float[] data, float y)
        {
            TruthLabel = new float[] { y};
            Data = data;
            LabelName = null;
        }
    }
}

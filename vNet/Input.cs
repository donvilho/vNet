using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    [Serializable]
    class Input
    {
        public float[] Y { get; private set; }
        public float[] Data { get; private set; }
        public string LabelName { get; private set; }


        public Input(float[] y, float[] data, string labelname)
        {
            Y = y;
            LabelName = labelname;
            Data = data;
        }
       
    }
}

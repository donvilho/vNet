using System;

namespace vNet
{
    [Serializable]
    public struct Input
    {
        public double[] TruthLabel { get; set; }
        public double[] Data { get; set; }
        public string LabelName { get; set; }
        public string Path { get; set; }

        public Input(double[] data, double[] y, string labelname, string path)
        {
            TruthLabel = y;
            LabelName = labelname;
            Data = data;
            Path = path;
        }

        public Input(double[] data, double[] y)
        {
            TruthLabel = y;
            Data = data;
            LabelName = null;
            Path = null;
        }

        public Input(double[] data, double y, string labelname)
        {
            TruthLabel = new double[] { y };
            Data = data;
            LabelName = labelname;
            Path = null;
        }

        public Input(double[] data, double y)
        {
            TruthLabel = new double[] { y };
            Data = data;
            LabelName = null;
            Path = null;
        }
    }
}
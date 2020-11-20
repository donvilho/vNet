namespace vNet
{
    internal struct Input
    {
        public float[] TruthLabel { get; set; }
        public float[] Data { get; set; }
        public string LabelName { get; set; }

        public Input(float[] data, float[] y, string labelname)
        {
            TruthLabel = y;
            LabelName = labelname;
            Data = data;
        }

        public Input(float[] data, float[] y)
        {
            TruthLabel = y;
            Data = data;
            LabelName = null;
        }

        public Input(float[] data, float y, string labelname)
        {
            TruthLabel = new float[] { y };
            Data = data;
            LabelName = labelname;
        }

        public Input(float[] data, float y)
        {
            TruthLabel = new float[] { y };
            Data = data;
            LabelName = null;
        }
    }
}
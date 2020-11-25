namespace vNet
{
    public struct Input
    {
        public float[] TruthLabel { get; set; }
        public float[] Data { get; set; }
        public string LabelName { get; set; }
        public string Path { get; set; }

        public Input(float[] data, float[] y, string labelname, string path)
        {
            TruthLabel = y;
            LabelName = labelname;
            Data = data;
            Path = path;
        }

        public Input(float[] data, float[] y)
        {
            TruthLabel = y;
            Data = data;
            LabelName = null;
            Path = null;
        }

        public Input(float[] data, float y, string labelname)
        {
            TruthLabel = new float[] { y };
            Data = data;
            LabelName = labelname;
            Path = null;
        }

        public Input(float[] data, float y)
        {
            TruthLabel = new float[] { y };
            Data = data;
            LabelName = null;
            Path = null;
        }
    }
}
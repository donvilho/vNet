using System.Numerics;

namespace vNet
{
    internal class ValidationInput
    {
        public dynamic[] Data { get; private set; }

        public ValidationInput()
        {
        }

        public ValidationInput(float[] data)
        {
            int n = 0;
            var offset = Vector<float>.Count;
            Data = new object[(data.Length / offset)];

            for (int i = 0; i < data.Length; i += offset)
            {
                Data[n] = new Vector<float>(data, i);
            }
        }
    }
}
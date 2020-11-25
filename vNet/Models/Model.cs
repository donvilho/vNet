namespace vNet
{
    public interface Model
    {
        int Classes { get; set; }
        float[] Output { get; set; }
        Activation activation { get; set; }
        Loss loss { get; set; }

        Dataset Data { get; set; }
        Neuron[] Neurons { get; set; }
    }
}
namespace vNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var trainingset = Utils.DataArrayCreator(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\training");
            var testset = Utils.DataArrayCreator(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\testing");
            var Dataset = new Dataset(trainingset, testset);

            Dataset.Reduce(30);

            var Model = new LogisticRegression(Dataset, constInit: true);

            Model.TrainModel(epoch: 300, learningRate: 0.001f, momentum: 1f, miniBatch: 256);

            //Model.TrainModel(epoch: 300, learningRate: 0.01f, momentum: 1f, miniBatch: 256); //OK veto
        }
    }
}
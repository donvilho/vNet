namespace vNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var trainingset = Utils.DataArrayCreator(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\training");
            var testset = Utils.DataArrayCreator(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\testing");
            var Dataset = new Dataset(trainingset, testset);

            //Dataset.Reduce(30);

            var Model = new LogisticRegression(Dataset);
            Model.TrainModel(50, 0.1f);
        }
    }
}
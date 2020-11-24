using ScottPlot;
using System.Diagnostics;

namespace vNet
{
    internal class Plot
    {
        /// <summary>
        /// Method for plotting data to .png
        /// </summary>
        /// <param name="dataPoints"></param>
        public static void Graph(double[,] dataPoints, float lr, int mb, int highest)
        {
            double[] DataError = new double[dataPoints.GetUpperBound(0) + 1];
            double[] DataAccuracy = new double[dataPoints.GetUpperBound(0) + 1];
            double[] dataXs = new double[dataPoints.GetUpperBound(0) + 1];
            double[] TrainingLoss = new double[dataPoints.GetUpperBound(0) + 1];

            for (int i = 0; i < dataXs.Length; i++)
            {
                dataXs[i] = i;
                DataError[i] = dataPoints[i, 0];
                DataAccuracy[i] = dataPoints[i, 1];
                TrainingLoss[i] = dataPoints[i, 2];
            }

            /// plot the data
            ///

            var plt = new ScottPlot.Plot(1024, 768);

            plt.PlotScatterHighlight(dataXs, DataAccuracy, label: "Accuracy");
            plt.PlotScatterHighlight(dataXs, DataError, label: "Test.Loss");
            plt.PlotScatterHighlight(dataXs, TrainingLoss, label: "Train.Loss");

            //plt.Grid(xSpacing:, ySpacing: .05);

            //plt.AxisAutoY(0.1);

            plt.Legend(location: legendLocation.middleLeft);

            plt.Title("Network lr: " + lr + " mb: " + mb);

            plt.XLabel("Epoch");
            plt.YLabel("Accuracy");

            plt.PlotVLine(highest, lineStyle: LineStyle.Dash, label: DataAccuracy[highest].ToString());

            plt.SaveFig("NN_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png");

            Process.Start(new ProcessStartInfo("NN_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png") { UseShellExecute = true });
        }
    }
}
using ScottPlot;
using System;
using System.Collections.Generic;
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

            var plt = new ScottPlot.Plot(1600, 900);

            plt.PlotScatterHighlight(dataXs, DataAccuracy, label: "Accuracy");
            plt.PlotScatterHighlight(dataXs, DataError, label: "Test.Loss");
            plt.PlotScatterHighlight(dataXs, TrainingLoss, label: "Train.Loss");

            //plt.Grid(xSpacing:, ySpacing: .05);

            plt.Legend(location: legendLocation.middleLeft);

            plt.Title("Network lr: " + lr + " mb: " + mb);

            plt.XLabel("Epoch");
            plt.YLabel("Accuracy");

            plt.PlotVLine(highest, lineStyle: LineStyle.Dash, label: DataAccuracy[highest].ToString());

            plt.SaveFig("NN_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png");

            Process.Start(new ProcessStartInfo("NN_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png") { UseShellExecute = true });
        }

        public static void GraphList(List<(double[,], int, float, int)> dataPoints)
        {
            var plt = new ScottPlot.Plot(1024, 768);

            foreach (var item in dataPoints)
            {
                double[] DataError = new double[item.Item1.GetUpperBound(0) + 1];
                double[] DataAccuracy = new double[item.Item1.GetUpperBound(0) + 1];
                double[] dataXs = new double[item.Item1.GetUpperBound(0) + 1];
                double[] TrainingLoss = new double[item.Item1.GetUpperBound(0) + 1];

                for (int i = 0; i < dataXs.Length; i++)
                {
                    dataXs[i] = i;
                    DataError[i] = item.Item1[i, 0];
                    DataAccuracy[i] = item.Item1[i, 1];
                    TrainingLoss[i] = item.Item1[i, 2];
                }

                plt.PlotScatterHighlight(dataXs, DataAccuracy, label: "lr:" + item.Item3 + " bat:" + item.Item4);
                //plt.PlotScatterHighlight(dataXs, DataError, label: item.Item3 + " Test.Loss");
                //plt.PlotScatterHighlight(dataXs, TrainingLoss, label: item.Item3 + " Train.Loss");
                plt.PlotVLine(item.Item2, lineStyle: LineStyle.Solid, lineWidth: 2, label: "lr:" + item.Item3 + " bat:" + item.Item4 + " Accuracy:" + Math.Round(DataAccuracy[item.Item2], 2).ToString());
            }

            /// plot the data
            ///

            //plt.Grid(xSpacing:, ySpacing: .05);

            //plt.AxisAutoY(0.1);
            plt.Axis(-30);

            plt.Legend(location: legendLocation.middleLeft, fontSize: 15);

            plt.Title("Network");

            plt.XLabel("Epoch");
            plt.YLabel("Accuracy");

            plt.SaveFig("NN.png");

            Process.Start(new ProcessStartInfo("NN.png") { UseShellExecute = true });
        }
    }
}
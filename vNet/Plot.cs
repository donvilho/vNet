using ScottPlot;
using ScottPlot.Drawing;
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
        public static void Graph(double[,] dataPoints, double lr, int mb, int highest)
        {
            double[] DataError = new double[dataPoints.GetUpperBound(0) + 1];
            double[] DataAccuracy = new double[dataPoints.GetUpperBound(0) + 1];
            double[] dataXs = new double[dataPoints.GetUpperBound(0) + 1];
            double[] TrainingLoss = new double[dataPoints.GetUpperBound(0) + 1];
            double[] Time = new double[dataPoints.GetUpperBound(0) + 1];

            for (int i = 0; i < dataXs.Length; i++)
            {
                dataXs[i] = i;
                DataError[i] = dataPoints[i, 0];
                DataAccuracy[i] = dataPoints[i, 1];
                TrainingLoss[i] = dataPoints[i, 2];
                Time[i] = dataPoints[i, 3];
            }

            // Data plot
            var plt = new ScottPlot.Plot(800, 600);
            plt.Style(Style.Gray1);
            plt.Colorset(Colorset.OneHalfDark);
            plt.PlotScatterHighlight(dataXs, DataAccuracy, label: "Test.Accuracy");
            plt.PlotScatterHighlight(dataXs, DataError, label: "Test.Loss");
            plt.PlotScatterHighlight(dataXs, TrainingLoss, label: "Train.Accuracy");
            plt.Legend(location: legendLocation.middleLeft, fontSize: 12);
            plt.Title("Network lr: " + lr + " mb: " + mb);
            plt.XLabel("Epoch");
            plt.YLabel("Accuracy");
            plt.Axis(-13);
            plt.PlotVLine(highest, lineStyle: LineStyle.Dash, label: Math.Round(DataAccuracy[highest], 3).ToString());
            plt.SaveFig("NN_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png");

            //Time plot
            var pltTime = new ScottPlot.Plot(800, 600);
            pltTime.Style(Style.Gray1);
            pltTime.Colorset(Colorset.OneHalfDark);
            pltTime.PlotScatterHighlight(dataXs, Time, label: "training time");
            pltTime.XLabel("Epoch", fontSize: 30);
            pltTime.YLabel("Seconds", fontSize: 30);
            pltTime.SaveFig("NN_Time_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png");

            // Open open plot images
            Process.Start(new ProcessStartInfo("NN_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png") { UseShellExecute = true });
            //Process.Start(new ProcessStartInfo("NN_Time_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png") { UseShellExecute = true });
        }

        public static void GraphList(List<(double, double[,], double, double, int, int, bool)> dataPoints)
        {
            var plt = new ScottPlot.Plot(800, 600);

            foreach (var item in dataPoints)
            {
                double[] DataError = new double[item.Item2.GetUpperBound(0) + 1];
                double[] DataAccuracy = new double[item.Item2.GetUpperBound(0) + 1];
                double[] dataXs = new double[item.Item2.GetUpperBound(0) + 1];
                double[] TrainingLoss = new double[item.Item2.GetUpperBound(0) + 1];

                for (int i = 0; i < dataXs.Length; i++)
                {
                    dataXs[i] = i;
                    DataError[i] = item.Item2[i, 0];
                    DataAccuracy[i] = item.Item2[i, 1];
                    TrainingLoss[i] = item.Item2[i, 2];
                }

                plt.PlotScatterHighlight(dataXs, DataAccuracy, label: "lr:" + item.Item3 + " bat:" + item.Item5);
                //plt.PlotScatterHighlight(dataXs, DataError, label: item.Item3 + " Test.Loss");
                //plt.PlotScatterHighlight(dataXs, TrainingLoss, label: item.Item3 + " Train.Loss");
                //plt.PlotVLine(item.Item2, lineStyle: LineStyle.Solid, lineWidth: 2, label: "lr:" + item.Item3 + " bat:" + item.Item4 + " Accuracy:" + Math.Round(DataAccuracy[item.Item2], 4).ToString());
            }

            /// plot the data
            ///

            //plt.Grid(xSpacing:, ySpacing: .05);

            //plt.AxisAutoY(0.1);
            //plt.Axis(50);

            plt.Legend(location: legendLocation.middleRight);

            plt.Title("Network");

            plt.XLabel("Epoch");
            plt.YLabel("Accuracy");
            plt.Style(Style.Gray1);
            plt.Colorset(Colorset.OneHalfDark);
            plt.SaveFig("NN.png");

            Process.Start(new ProcessStartInfo("NN.png") { UseShellExecute = true });
        }
    }
}
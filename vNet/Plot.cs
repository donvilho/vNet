using ScottPlot;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    class Plot
    {

        /// <summary>
        /// Method for plotting data to .png
        /// </summary>
        /// <param name="dataPoints"></param>
        public static void Graph(double[,] dataPoints)
        {

            double[] DataError = new double[dataPoints.GetUpperBound(0) + 1];
            double[] DataAccuracy = new double[dataPoints.GetUpperBound(0) + 1];
            double[] dataXs = new double[dataPoints.GetUpperBound(0) + 1];

            for (int i = 0; i < dataXs.Length; i++)
            {
                dataXs[i] = i;
                DataError[i] = dataPoints[i, 0];
                DataAccuracy[i] = dataPoints[i, 1];
            }

            /// plot the data
            /// 
          
            var plt_acc = new ScottPlot.Plot(800, 600);
            var plt_loss = new ScottPlot.Plot(800, 600);
            plt_acc.PlotScatter(dataXs, DataAccuracy, label: "Accuracy");
            plt_loss.PlotScatter(dataXs, DataError, label: "Error rate");

            plt_acc.Grid(xSpacing: 20, ySpacing: .1);
            plt_loss.Grid(xSpacing: 5, ySpacing: .1);

            plt_acc.Legend(location: legendLocation.upperLeft);
            plt_loss.Legend(location: legendLocation.upperLeft);

            plt_acc.Title("Accuracy");
            plt_loss.Title("Loss");

            plt_acc.XLabel("Epoch");
            plt_acc.YLabel("Accuracy");

            plt_loss.XLabel("Epoch");
            plt_loss.YLabel("Loss");

            plt_acc.SaveFig("NN_acc.png");
            plt_loss.SaveFig("NN_loss.png");
            Process.Start(new ProcessStartInfo(@"NN_acc.png") { UseShellExecute = true });
            Process.Start(new ProcessStartInfo(@"NN_loss.png") { UseShellExecute = true });
        }
    }
}

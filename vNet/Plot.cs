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
        public static void Graph(double[,] dataPoints,float lr, int mb)
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
          
            var plt = new ScottPlot.Plot(800, 600);
          
            plt.PlotScatter(dataXs, DataAccuracy, label: "Accuracy");
            plt.PlotScatter(dataXs, DataError, label: "Error rate");

            //plt.Grid(xSpacing: 20, ySpacing: .1);

            plt.AxisAutoY(0.1);

            plt.Legend(location: legendLocation.upperLeft);


            plt.Title("Network lr: "+lr+" mb: "+mb);
       

            plt.XLabel("Epoch");
            plt.YLabel("Accuracy");

             

     
            plt.SaveFig("NN_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png");

            Process.Start(new ProcessStartInfo("NN_plot_lr_" + lr.ToString() + "_batchsize_" + mb.ToString() + ".png") { UseShellExecute = true });
        }
    }
}

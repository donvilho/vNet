using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    internal static class Operations
    {
        public static double[,,] MaxPool(double[,,] x, int dim, int stride)
        {
            int dimLen = (int)(x.GetUpperBound(0) + 1) - dim / stride + 1;

            return null;
        }

        public static double[,,] Convolution2DF(double[,] x, double[,,] k, int stride, int padding)
        {
            // Output height = (Input height + padding height top +padding height bottom -kernel height) / (stride height) +1.
            // Output width = (Output width + padding width right +padding width left -kernel width) / (stride width) +1.

            int dimLen = (int)((x.GetUpperBound(0) + 1) + (2 * padding) - (k.GetUpperBound(0) + 1)) / stride + 1;

            var result = new double[dimLen, dimLen, k.GetUpperBound(2) + 1];

            var xOffset = padding;
            var yOffset = padding;

            var xLim = x.GetUpperBound(0) + 1;

            // for result lenght
            for (int i = 0; i < dimLen; i++) // + stride
            {
                for (int j = 0; j < dimLen; j++)
                {
                    for (int z = 0; z < k.GetUpperBound(2) + 1; z++)
                    {
                        for (int xi = -padding; xi < xLim; xi++)
                        {
                            for (int xj = -padding; xj < xLim; xj++)
                            {
                                //for k lenght
                                for (int ki = 0; ki < k.GetUpperBound(0) + 1; ki++)
                                {
                                    for (int kj = 0; kj < k.GetUpperBound(1) + 1; kj++)
                                    {
                                        if (xi + ki < xLim && xj + kj < xLim)
                                        {
                                            if (xi + ki >= 0 && xj + kj >= 0)
                                            {
                                                result[i, j, z] += x[xi + ki, xj + kj] * k[ki, kj, z];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }

        public static double[,,] Convolution2D(double[,] x, double[,,] k, int stride, int padding)
        {
            // Output height = (Input height + padding height top +padding height bottom -kernel height) / (stride height) +1.
            // Output width = (Output width + padding width right +padding width left -kernel width) / (stride width) +1.

            int dimLen = (int)((x.GetUpperBound(0) + 1) + (2 * padding) - (k.GetUpperBound(0) + 1)) / stride + 1;

            var result = new double[dimLen, dimLen, k.GetUpperBound(2) + 1];

            if (padding > 0)
            {
                x = ApplyPadding(x, padding);
            }

            //Parallel.For(0, (x.GetUpperBound(0) + 1 - k.GetUpperBound(0)), i =>
            //{
            // for x lenght
            for (int i = 0; i < (x.GetUpperBound(0) + 1 - k.GetUpperBound(0)); i++) // + stride
            {
                for (int j = 0; j < (x.GetUpperBound(1) + 1 - k.GetUpperBound(1)); j++)
                {
                    //for k lenght
                    for (int ki = 0; ki < k.GetUpperBound(0) + 1; ki++)
                    {
                        for (int kj = 0; kj < k.GetUpperBound(1) + 1; kj++)
                        {
                            for (int z = 0; z < k.GetUpperBound(2) + 1; z++)
                            {
                                result[i, j, z] += x[i + ki, j + kj] * k[ki, kj, z];
                            }
                        }
                    }
                }
            }
            //});

            return result;
        }

        public static double[,] Convolution2D(double[,] x, double[,] k, int stride, int padding)
        {
            // Output height = (Input height + padding height top +padding height bottom -kernel height) / (stride height) +1.
            // Output width = (Output width + padding width right +padding width left -kernel width) / (stride width) +1.

            int dimLen = (int)((x.GetUpperBound(0) + 1) + (2 * padding) - (k.GetUpperBound(0) + 1)) / stride + 1;

            var result = new double[dimLen, dimLen];

            if (padding > 0)
            {
                x = ApplyPadding(x, padding);
            }

            //Console.WriteLine(x.GetUpperBound(0));

            // for x lenght
            for (int i = 0; i < x.GetUpperBound(0); i++) // + stride
            {
                for (int j = 0; j < x.GetUpperBound(1); j++)
                {
                    //Console.WriteLine(x[i, j]);
                    //for k lenght
                    for (int ki = 0; ki < k.GetUpperBound(0) + 1; ki++)
                    {
                        for (int kj = 0; kj < k.GetUpperBound(1) + 1; kj++)
                        {
                            result[i, j] += x[i + ki, j + kj] * k[ki, kj];
                            //      Console.WriteLine(x[i + ki, j + kj] + "*" + k[ki, kj]);
                        }
                    }
                    // Console.WriteLine(result[i, j]);
                }
            }

            return result;
        }

        public static double[,] ApplyPadding(double[,] x, int padding)
        {
            var result = new double[x.GetUpperBound(0) + 1 + (2 * padding), x.GetUpperBound(1) + 1 + (2 * padding)];

            for (int i = 0; i < x.GetUpperBound(0) + 1; i++)
            {
                for (int j = 0; j < x.GetUpperBound(1) + 1; j++)
                {
                    result[(i + padding), (j + padding)] = x[i, j];
                }
            }

            return result;
        }
    }
}
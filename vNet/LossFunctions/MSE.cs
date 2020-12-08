using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet.LossFunctions
{
    internal class MSE : Loss
    {
        public override double Calculate(double[] n, double[] t)
        {
            var result = 0d;
            for (int i = 0; i < n.Length; i++)
            {
                result += n[i] - t[i];
            }
            return result * result;
        }
    }
}
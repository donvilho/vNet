using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet.LossFunctions
{
    internal class CrossEntropy : Loss
    {
        public override double Calculate(double[] n, double[] t)
        {
            var loss = 0d;
            for (int i = 0; i < n.Length; i++)
            {
                loss += t[i] * Math.Log(n[i]) - ((1 - t[i]) * Math.Log(1 - n[i]));
            }

            //return (double)-loss / n.Length;
            return (double)-loss / n.Length;
        }
    }
}
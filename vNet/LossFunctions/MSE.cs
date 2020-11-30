using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet.LossFunctions
{
    internal class MSE : Loss
    {
        public override float Calculate(float[] n, float[] t)
        {
            var result = 0f;
            for (int i = 0; i < n.Length; i++)
            {
                result += n[i] - t[i];
            }
            return result * result;
        }
    }
}
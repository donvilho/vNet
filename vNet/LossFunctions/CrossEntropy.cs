using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet.LossFunctions
{
    internal class CrossEntropy : Loss
    {
        public override float Calculate(float[] n, float[] t)
        {
            var loss = 0d;
            for (int i = 0; i < n.Length; i++)
            {
                loss += -t[i] * Math.Log(n[i]) - ((1 - t[i]) * Math.Log(1 - n[i])); // tätä muutettiin
                //loss += -Data.TruthLabel[i] * Math.Log(Net.Error[i]);
            }

            return (float)-loss / n.Length;
        }
    }
}
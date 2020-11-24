using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet.Regularization
{
    internal class L2 : Reg
    {
        public override float CalcReg(float[] Neurons)
        {
            return (float)Math.Sqrt(Neurons.Sum());
        }
    }
}
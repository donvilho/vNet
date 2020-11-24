using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet.Regularization
{
    internal abstract class Reg
    {
        public abstract float CalcReg(float[] Neurons);
    }
}
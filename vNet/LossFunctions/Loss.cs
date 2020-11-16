using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    internal abstract class Loss
    {
        public abstract float Calculate(float[] n, float[] t);
    }
}
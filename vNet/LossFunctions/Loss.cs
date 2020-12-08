using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    public abstract class Loss
    {
        public abstract double Calculate(double[] n, double[] t);
    }
}
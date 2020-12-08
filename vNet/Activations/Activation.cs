using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    public abstract class Activation
    {
        public abstract double Activate(double n);

        public abstract double[] Activate(double[] n);

        public abstract double[] Activate(Neuron[] n);

        public abstract double Derivate(double n, double t);

        public abstract int Compare(double[] n, double[] t);
    }
}
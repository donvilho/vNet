using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    public abstract class Activation
    {
        public abstract float Activate(float n);

        public abstract float[] Activate(float[] n);

        public abstract float[] Activate(Neuron[] n);

        public abstract float Derivate(float n, float t);

        public abstract int Compare(float[] n, float[] t);
    }
}
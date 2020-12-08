using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using vNet.Activations;

namespace vNet.Activations
{
    internal class Sigmoid : Activation
    {
        public override double[] Activate(double[] Neurons)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = Utils.SigmoidNormal(Neurons[i]);
            }

            return Neurons;
        }

        public override double Activate(double n)
        {
            return Utils.SigmoidNormal(n);
        }

        public override double[] Activate(Neuron[] n)
        {
            var res = new double[n.Length];

            for (int i = 0; i < n.Length; i++)
            {
                res[i] = Utils.SigmoidNormal(n[i].Z);
            }

            return res;
        }

        public override int Compare(double[] n, double[] t)
        {
            for (int i = 0; i < n.Length; i++)
            {
                n[i] = (double)Math.Round(n[i]);
            }

            return n.SequenceEqual(t) ? 1 : 0;
        }

        public override double Derivate(double n, double t)
        {
            return Utils.SigmoidNormalDerivate(n) * (n - t);
        }
    }
}
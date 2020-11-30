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
        public override float[] Activate(float[] Neurons)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = Utils.SigmoidNormal(Neurons[i]);
            }

            return Neurons;
        }

        public override float Activate(float n)
        {
            return Utils.SigmoidNormal(n);
        }

        public override float[] Activate(Neuron[] n)
        {
            var res = new float[n.Length];

            for (int i = 0; i < n.Length; i++)
            {
                res[i] = Utils.SigmoidNormal(n[i].Z);
            }

            return res;
        }

        public override int Compare(float[] n, float[] t)
        {
            for (int i = 0; i < n.Length; i++)
            {
                n[i] = (float)Math.Round(n[i]);
            }

            return n.SequenceEqual(t) ? 1 : 0;
        }

        public override float Derivate(float n, float t)
        {
            return Utils.SigmoidNormalDerivate(n) * (n - t);
        }
    }
}
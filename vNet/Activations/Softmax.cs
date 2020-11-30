using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet.Activations
{
    internal class Softmax : Activation
    {
        public override float[] Activate(float[] Neurons)
        {
            var ExpSum = Neurons.Sum();

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] /= ExpSum;
            }

            return Neurons;
        }

        public override float Activate(float n)
        {
            throw new NotImplementedException();
        }

        public override float[] Activate(Neuron[] n)
        {
            var result = new float[n.Length];

            var expSum = 0f;

            for (int i = 0; i < n.Length; i++)
            {
                n[i].A = (float)Utils.exp4(n[i].Z);
                expSum += n[i].A;
            }

            for (int i = 0; i < n.Length; i++)
            {
                result[i] = n[i].A / expSum;
            }

            return result;
        }

        public override int Compare(float[] n, float[] t)
        {
            return (n.ToList().IndexOf(n.Max()) == t.ToList().IndexOf(t.Max()) ? 1 : 0);
        }

        public override float Derivate(float n, float t)
        {
            /*
            if (t == 1)
            {
                return n - 1;
            }
            else
            {
                return n;
            }
            */

            return n - t;
        }
    }
}
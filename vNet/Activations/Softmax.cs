using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet.Activations
{
    internal class Softmax : Activation
    {
        public override double[] Activate(double[] Neurons)
        {
            var ExpSum = Neurons.Sum();

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] /= ExpSum;
            }

            return Neurons;
        }

        public override double Activate(double n)
        {
            throw new NotImplementedException();
        }

        public override double[] Activate(Neuron[] n)
        {
            var result = new double[n.Length];

            var expSum = 0d;

            for (int i = 0; i < n.Length; i++)
            {
                n[i].A = (double)Math.Exp(n[i].Z);
                expSum += n[i].A;
            }

            for (int i = 0; i < n.Length; i++)
            {
                result[i] = n[i].A / expSum;
            }

            return result;
        }

        public override int Compare(double[] n, double[] t)
        {
            return (n.ToList().IndexOf(n.Max()) == t.ToList().IndexOf(t.Max()) ? 1 : 0);
        }

        public override double Derivate(double n, double t)
        {
            if (t == 1)
            {
                return n - 1;
            }
            else
            {
                return n;
            }

            //return n - t;
        }
    }
}
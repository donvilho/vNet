using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    public class TrainingSetup
    {
        public float[] learningrates;
        public int[] batches;

        public TrainingSetup(float[] lrs, int[] bts) => (learningrates, batches) = (lrs, bts);
    }
}
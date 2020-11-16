namespace vNet
{
    internal class mics
    {
        /*
     public float Forward((float[], float[], string) input)
     {
         var ExpSum = 0f;

         for (int i = 0; i < Neurons.Length; i++)
         {
             Neurons[i] = Bias[i] + Utils.Dot(Weights[i], input.Item1);
             //Calc EXP SUM
             ExpSum += (float)Math.Exp(Neurons[i]);
         }

         return ExpSum;
     }
     */

        /*
        public float Backward((float[], float[], string) input, float ExpSum, bool ParallelDegree = false)
        {
            var Loss = 0f;

            switch (ParallelDegree)
            {
                case false:
                    for (int i = 0; i < Derivate.Length; i++)
                    {
                        //CalcError/activate

                        Error[i] = (float)Math.Exp(Neurons[i]) / ExpSum;
                        Loss += input.Item2[i] * (float)Math.Log(Error[i]);

                        //Loss += input.Item2[i] * (float)Math.Log(Math.Exp(Neurons[i]) / ExpSum);
                        //CalcDerivates
                        //D-A
                        Derivate[i] = Error[i] - input.Item2[i];
                        //D-Z
                        Derivate[i] *= Error[i] * (1 - Error[i]);

                        for (int j = 0; j < WeightCache[i].Length; j++)
                        {
                            //D-W
                            WeightCache[i][j] += input.Item1[j] * Derivate[i];
                            //D-B
                            BiasCache[i] += Bias[i] * Derivate[i];
                        }
                    }
                    break;

                case true:

                    void Kernel(int i)
                    {
                        Error[i] = (float)Math.Exp(Neurons[i]) / ExpSum;
                        Loss += input.Item2[i] * (float)Math.Log(Error[i]);

                        Derivate[i] = (Error[i] - input.Item2[i]) * Error[i] * (1 - Error[i]);
                        //Derivate[i] *= Error[i] * (1 - Error[i]);

                        for (int j = 0; j < WeightCache[i].Length; j++)
                        {
                            //D-W
                            WeightCache[i][j] += input.Item1[j] * Derivate[i];
                            //D-B
                            BiasCache[i] += Bias[i] * Derivate[i];
                        }
                    }

                    Parallel.For(0, Derivate.Length, Kernel);
                    break;
            }

            return Loss;
        }
        */
    }
}
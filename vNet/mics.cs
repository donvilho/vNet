namespace vNet
{
    internal class mics
    {
        /*
          var timer = new Stopwatch();

            var trSet = Utils.CreateDataMatrix(@"C:\Users\ville\Downloads\mnist_png.tar\mnist_png\training", 50);

            Utils.ShuffleDataMatrix(trSet);

            var Weights = Utils.GenerateMatrix(trSet.Item1.GetLength(1), trSet.Item2.GetLength(1));
            var WeightsCache = new float[trSet.Item1.GetLength(0), trSet.Item1.GetLength(1), trSet.Item2.GetLength(1)];

            var Batch = new float[trSet.Item1.GetLength(0), trSet.Item2.GetLength(1)];
            var BatchTruth = new float[trSet.Item2.GetLength(0), trSet.Item1.GetLength(1)];

            for (int e = 0; e < 200; e++)
            {
                Console.WriteLine(e);

                // Forward

                var partition = Partitioner.Create(0, Batch.GetLength(0));
                var expsum = 0f;
                timer.Restart();

                for (int i = 0; i < Batch.GetLength(0); i++)
                {
                    for (int j = 0; j < Weights.GetLength(1); j++) // Neuron count
                    {
                        Batch[i, j] = 0; // ??

                        for (int k = 0; k < Weights.GetLength(0); k++) // image data count
                        {
                            Batch[i, j] += trSet.Item1[i, k] * Weights[k, j];
                            BatchTruth[i, j] = trSet.Item2[i, j];
                        }

                        // Summ exps
                        var x = Batch[i, j];
                        Batch[i, j] = (720 + x * (720 + x * (360 + x * (120 + x * (30 + x * (6 + x)))))) * 0.0013888888f;
                        expsum += Batch[i, j];
                    }

                    // softmax
                    for (int j = 0; j < Batch.GetLength(1); j++)
                    {
                        Batch[i, j] /= expsum;
                    }

                    for (int j = 0; j < Batch.GetLength(1); j++)
                    {
                        Batch[i, j] -= BatchTruth[i, j];

                        for (int k = 0; k < Weights.GetLength(0); k++)
                        {
                            WeightsCache[i, k, j] -= (trSet.Item1[j, k] * Batch[i, j]) * 0.1f;
                        }
                    }
                }

                Console.WriteLine(timer.ElapsedMilliseconds);
                Console.ReadKey();

                var part = Partitioner.Create(0, WeightsCache.GetLength(2));

                Parallel.ForEach(part, range =>
                {
                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        for (int j = 0; j < WeightsCache.GetLength(0); j++)
                        {
                            for (int k = 0; k < WeightsCache.GetLength(1); k++)
                            {
                                Weights[k, i] -= WeightsCache[j, k, i];
                                WeightsCache[j, k, i] = 0;
                            }
                        }
                    }
                });

                timer.Stop();
                Console.WriteLine(timer.ElapsedMilliseconds);
            }

            Console.WriteLine("Done");
        */

        /*
        private var timer = new Stopwatch();

        private var a = Utils.GenerateMatrix(10, 800);
        private var b = Utils.GenerateMatrix(1000, 800);

        private var partition = Partitioner.Create(0, 100);

        private var c = Utils.GenerateMatrix(100, 10, setNumber: true);

        timer.Restart();
            for (int e = 0; e< 1000; e++)
            {
                Parallel.ForEach(partition, value =>
                {
                    for (int i = value.Item1; i<value.Item2; i++)
                    {
                        for (int j = 0; j< 10; j++)
                        {
                            for (int k = 0; k< 800; k++)
                            {
                                c[i, j] += a[j, k] * b[i, k];
                            }
}
                    }
                });
            }
            timer.Stop();
Console.WriteLine("foreach: " + timer.ElapsedMilliseconds);
Array.Clear(c, 0, c.Length);
timer.Restart();
for (int e = 0; e < 1000; e++)
{
    Parallel.For(0, 100, i =>
    {
        for (int j = 0; j < 10; j++)
        {
            for (int k = 0; k < 800; k++)
            {
                c[i, j] += a[j, k] * b[i, k];
            }
        }
    });
}
timer.Stop();
Console.WriteLine("for: " + timer.ElapsedMilliseconds);

Array.Clear(c, 0, c.Length);
timer.Restart();

for (int e = 0; e < 1000; e++)
{
    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            for (int k = 0; k < 800; k++)
            {
                c[i, j] += a[j, k] * b[i, k];
            }
        }
    }
}

timer.Stop();
Console.WriteLine("Sequential: " + timer.ElapsedMilliseconds);
        */
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
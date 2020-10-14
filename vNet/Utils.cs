using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace vNet
{
    class Utils
    {

        public static float[] VectorScalarMultiply(float[]A , float B)
        {
            for (int i = 0; i < A.Length; i++)
            {
                A[i] *= B;
            }
            return A;
        }

        public static float[] CalculateError(float[] NetOut, float[] Truth)
        {
            float[] Error = new float[NetOut.Length];

            if(NetOut.Length == Truth.Length)
            {
                for (int i = 0; i < NetOut.Length; i++)
                {
                    Error[i] = NetOut[i] - Truth[i];
                }
            }

            return Error;
        }

        public static float[] Sigmoid_Derivate(float[] value)
        {

            for (int i = 0; i < value.Length; i++)
            {
                value[i] = (float)(1 / (1 + Math.Exp(-value[i])) * (1 - (1 / (1 + Math.Exp(-value[i])))));
            }

            return value;
        }

        public static float[] Generate_Vector(int size, double min = 0.1, double max = 0.9)
        {

            /// super randomizer
            /// järkyttävä overkill mutta olkoot

            RNGCryptoServiceProvider random = new RNGCryptoServiceProvider();
            var Bytes = new byte[4];
            random.GetBytes(Bytes);

            RNGCryptoServiceProvider random1 = new RNGCryptoServiceProvider();
            var Bytes1 = new byte[4];
            random1.GetBytes(Bytes1);

            RNGCryptoServiceProvider random2 = new RNGCryptoServiceProvider();
            var Bytes2 = new byte[4];
            random2.GetBytes(Bytes2);

            Random rand = new Random(BitConverter.ToInt32(Bytes, 0) + BitConverter.ToInt32(Bytes1, 0) - BitConverter.ToInt32(Bytes2, 0));

            float[] Result = new float[size];
            for (int i = 0; i < size; ++i)
            {
                Result[i] = Convert.ToSingle((rand.NextDouble() * max) - (min));
            }
            random.Dispose();
            random1.Dispose();
            random2.Dispose();
            return Result;
        }

        public static float[] LabelVectorCreator(int Size, string Pos)
        {
            int.TryParse(Pos, out int num);
            var array = new float[Size];
            Array.Clear(array, 0, array.Length);
            array[num] = 1;
            return array;
        }

        public static Dataset DatasetCreator(string path)
        {

            Console.WriteLine("Creating dataset from files.. please wait, this may take few seconds");

            ConcurrentBag<Input> TrainingData = new ConcurrentBag<Input>();
            ConcurrentBag<Input> TestData = new ConcurrentBag<Input>();


            float[] truthLabel;


            try
            {
                var sets = Directory.GetDirectories(path);

                for (int s = 0; s < sets.Length; s++)
                {
                    string setName = new DirectoryInfo(sets[s]).Name;

                    var labels = Directory.GetDirectories(sets[s]);

                    for (int i = 0; i < labels.Length; i++)
                    {
                        Console.WriteLine(i);
                        string[] files = Directory.GetFiles(labels[i]);
                        string label = new DirectoryInfo(labels[i]).Name;
                  
                            Parallel.ForEach(files, (img) =>
                            {

                                if (labels.Length < 3)
                                {
                                    truthLabel = new float[1] { i };
                                }
                                else
                                {
                                    truthLabel = LabelVectorCreator(labels.Length, label);
                                }


                                if (setName.Contains("training"))
                                {
                                    TrainingData.Add(new Input(truthLabel, ImageToArray(img), label));
                                }
                                else
                                {
                                    TestData.Add(new Input(truthLabel, ImageToArray(img), label));
                                }
                            });
                    }
                }

                return new Dataset(TrainingData.ToArray(), TestData.ToArray());
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return new Dataset(TrainingData.ToArray(), TestData.ToArray());
            }
        }

        public static float[] ImageToArray(string Path)
        {
            Bitmap img = (Bitmap)Image.FromFile(Path);
            float[] Result = new float[img.Height * img.Width];
     
            for (int i = 0; i <= img.Height - 1; i++)
            {
                for (int j = 0; j <= img.Width - 1; j++)
                {
                    Color pixel = img.GetPixel(j, i);
                    float color = (pixel.R + pixel.B + pixel.G) / 3;
                    Result[img.Height * i + j] = color/255;
                    
                }
            }
            return Result;
        }

    }
}

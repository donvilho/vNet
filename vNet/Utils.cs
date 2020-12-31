using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Security.AccessControl;
using System.Security.Cryptography;
using System.Security.Principal;
using System.Threading.Tasks;

namespace vNet
{
    internal class Utils
    {
        public static Dataset DatasetFromBinary(string path, int DropoutLowerThreshold = 0, int DropoutUpperThreshold = 0)
        {
            try
            {
                Stream Reader = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
                IFormatter formatter = new BinaryFormatter();
                var dataset = (Dataset)formatter.Deserialize(Reader);
                Reader.Close();

                if (DropoutLowerThreshold != 0 | DropoutUpperThreshold != 0)
                {
                    var temp = new List<int>();
                    var interMidLayer = new double[dataset.InputLenght];

                    for (int i = 0; i < dataset.TrainingData.Length; i++)
                    {
                        for (int j = 0; j < dataset.InputLenght; j++)
                        {
                            interMidLayer[j] += dataset.TrainingData[i].Data[j] > 0 ? 1 : 0;
                        }
                    }

                    /*
                    for (int i = 0; i < interMidLayer.Length; i++)
                    {
                        interMidLayer[i] /= dataset.TrainingData.Length;
                    }
                    */

                    for (int i = 0; i < interMidLayer.Length; i++)
                    {
                        if (interMidLayer[i] > DropoutLowerThreshold | interMidLayer[i] < DropoutUpperThreshold)
                        {
                            temp.Add(i);
                        }
                    }

                    dataset.ApplyConnectionMask(temp.ToArray());
                }

                return dataset;
            }
            catch (Exception Ex)
            {
                Console.WriteLine(Ex.Message);
                return null;
            }
        }

        public static void ColoredConsoleWrite(ConsoleColor firstColor, string firstText)
        {
            Console.ForegroundColor = firstColor;
            Console.Write(firstText);
            Console.ResetColor();
        }

        public static void DatasetToBinary(Dataset dataset, string datasetName)
        {
            try
            {
                Stream writer = new FileStream(datasetName + ".bin", FileMode.Create, FileAccess.Write, FileShare.None);
                IFormatter formatter = new BinaryFormatter();
                formatter.Serialize(writer, dataset);
                writer.Close();
            }
            catch (Exception Ex)
            {
                Console.WriteLine(Ex.Message);
            }
        }

        public static double NumericSigmoid(double x)
        {
            return (double)(1 / (1 + Math.Pow(0.3678749025, x)));
        }

        public static double SigmoidNormalDerivate(double x)
        {
            return (double)(1 / (1 + Utils.exp1(-x)) * (1 - (1 / (1 + Utils.exp1(-x)))));
        }

        public static double SigmoidSuperDerivate(double x)
        {
            var r = (2 + 2 * (Math.Abs(x)));

            return (double)(2 / (r * r));
        }

        public static double SigmoidSuper(double x)
        {
            return (double)(x / (2 + 2 * Math.Abs(x)) + 0.5);
        }

        public static double SigmoidNormal(double A)
        {
            return (double)(1 / (1 + Math.Exp(-A)));
        }

        public static double SigmoidFast(double value)
        {
            double k = (double)Math.Exp(value);
            return k / (1.0f + k);
        }

        public static double SimdVectorProd(double[] left, double right)
        {
            var offset = Vector<double>.Count;
            double[] result = new double[left.Length];
            int i = 0;
            for (i = 0; i < left.Length; i += offset)
            {
                var v1 = new Vector<double>(left, i);
                (v1 * right).CopyTo(result, i);
            }

            //remaining items
            for (; i < left.Length; ++i)
            {
                result[i] = left[i] * right;
            }

            return result.Sum();
        }

        public static double exp1(double x)
        {
            return (6 + x * (6 + x * (3 + x))) * 0.16666666f;
        }

        public static double exp2(double x)
        {
            return (24 + x * (24 + x * (12 + x * (4 + x)))) * 0.041666666f;
        }

        public static double exp3(double x)
        {
            return (120 + x * (120 + x * (60 + x * (20 + x * (5 + x))))) * 0.0083333333f;
        }

        public static double exp4(double x)
        {
            return (720 + x * (720 + x * (360 + x * (120 + x * (30 + x * (6 + x)))))) * 0.0013888888f;
        }

        public static double exp5(double x)
        {
            return (5040 + x * (5040 + x * (2520 + x * (840 + x * (210 + x * (42 + x * (7 + x))))))) * 0.00019841269f;
        }

        public static double exp6(double x)
        {
            return (40320 + x * (40320 + x * (20160 + x * (6720 + x * (1680 + x * (336 + x * (56 + x * (8 + x)))))))) * 2.4801587301e-5;
        }

        public static double exp7(double x)
        {
            return (362880 + x * (362880 + x * (181440 + x * (60480 + x * (15120 + x * (3024 + x * (504 + x * (72 + x * (9 + x))))))))) * 2.75573192e-6;
        }

        public static void ShuffleDataMatrix((double[,], int[,]) Data)
        {
            var rand = new Random();
            for (int i = Data.Item1.GetLength(0) - 1; i > 1; i--)
            {
                int d = rand.Next(i + 1);

                for (int j = 0; j < Data.Item1.GetLength(1); j++)
                {
                    var value = Data.Item1[i, j];
                    Data.Item1[i, j] = Data.Item1[d, j];
                    Data.Item1[d, j] = value;
                }
            }
        }

        public static (double[,], int[,]) CreateDataMatrix(string path, int ReduceSizeTo = 100)
        {
            Console.WriteLine("Creating dataset from files.. please wait, this may take few seconds");

            var labels = Directory.GetDirectories(path);

            var allFiles = Directory.GetFiles(path, "*", SearchOption.AllDirectories);
            var imageSize = Image.FromFile(allFiles[0]);

            // Shuffle

            if (ReduceSizeTo < 100)
            {
                var rand = new Random();
                for (int Count = allFiles.Length - 1; Count > 1; Count--)
                {
                    int i = rand.Next(Count + 1);
                    var value = allFiles[i];
                    allFiles[i] = allFiles[Count];
                    allFiles[Count] = value;
                }

                var temp = allFiles.Take(allFiles.Length / 100 * ReduceSizeTo);

                allFiles = temp.ToArray();
            }

            var DataMatrix = new double[allFiles.Length, (imageSize.Width * imageSize.Height)];
            var TruthMatrix = new int[allFiles.Length, labels.Length];

            Parallel.For(0, DataMatrix.GetLength(0), i =>
            {
                TruthMatrix[i, int.Parse(new DirectoryInfo(allFiles[i]).Parent.Name)] = 1;

                var img = (Bitmap)Image.FromFile(allFiles[i]);

                for (int j = 0; j < img.Height; j++)
                {
                    for (int k = 0; k < img.Width; k++)
                    {
                        Color color = img.GetPixel(j, k);
                        DataMatrix[i, (j * img.Width) + k] = ((color.R + color.G + color.B) / 3) / 254;
                    }
                }

                img.Dispose();
            });

            return (DataMatrix, TruthMatrix);
        }

        public static Input[] DataArrayCreator(string path)
        {
            Console.WriteLine("Creating dataset from files.. please wait, this may take few seconds");

            ConcurrentBag<Input> Dataset = new ConcurrentBag<Input>();

            var labels = Directory.GetDirectories(path);

            for (int i = 0; i < labels.Length; i++)
            {
                // Console.WriteLine(i);
                string[] files = Directory.GetFiles(labels[i]);
                string labelName = new DirectoryInfo(labels[i]).Name;

                Parallel.ForEach(files, (img) =>
                {
                    if (labels.Length <= 2)
                    {
                        var truthLabel = i;
                        Dataset.Add(new Input(ImageToArray(img), truthLabel, labelName));
                    }
                    else
                    {
                        var truthLabel = LabelVectorCreator(labels.Length, i);
                        Dataset.Add(new Input(ImageToArray(img), truthLabel, labelName, img));
                    }
                });
            }

            return Dataset.ToArray();
        }

        public static List<Input> CSVtoArray(string path)
        {
            var result = new List<Input>();

            var lines = File.ReadAllLines(path);

            for (int i = 1; i < lines.Length; i++)
            {
                var line = lines[i].Split(',');
                var temp = new double[line.Length];

                for (int j = 0; j < line.Length; j++)
                {
                    temp[j] = double.Parse(line[j], CultureInfo.InvariantCulture.NumberFormat);
                }
                var data = new double[] { temp[1], temp[2] };

                result.Add(new Input(data, temp[3]));
            }

            return result;
        }

        public static double Dot(double[] a, double[] b)
        {
            var temp = 0d;
            for (int i = 0; i < b.Length; i++)
            {
                temp += a[i] * b[i];
            }
            return temp;
        }

        public static double[] Dot(double[,] a, double[] b)
        {
            var temp = new double[a.GetLength(0)];

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    temp[i] += a[i, j] * b[j];
                }
                //temp += a[i] * b[i];
            }
            return temp;
        }

        public static double[] Multiply2(double[] a, double[] b)
        {
            var result = new double[a.Length * b.Length];
            var c = 0;
            for (int i = 0; i < a.Length; i++)
            {
                for (int j = 0; j < b.Length; j++)
                {
                    result[c] = a[i] * b[j];
                    c++;
                }
            }

            return result;
        }

        public static double[] Multiply(double[] a, double[] b)
        {
            var B = b.Sum();

            for (int i = 0; i < b.Length; i++)
            {
                a[i] *= B;
            }

            return a;
        }

        public static double[] VectorScalarMultiply(double[] A, double B)
        {
            for (int i = 0; i < A.Length; i++)
            {
                A[i] *= B;
            }
            return A;
        }

        public static double[] CalculateError(double[] NetOut, double[] Truth)
        {
            double[] Error = new double[NetOut.Length];

            if (NetOut.Length == Truth.Length)
            {
                for (int i = 0; i < NetOut.Length; i++)
                {
                    Error[i] = NetOut[i] - Truth[i];
                }
            }

            return Error;
        }

        public static double[] Sigmoid_Derivate(double[] value)
        {
            for (int i = 0; i < value.Length; i++)
            {
                value[i] = (double)(1 / (1 + Math.Exp(-value[i])) * (1 - (1 / (1 + Math.Exp(-value[i])))));
            }

            return value;
        }

        public static double[] Generate_Vector(int size, double min = -0.01, double max = 0.01, double number = 0)
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

            var Result = new double[size];

            for (int i = 0; i < size; ++i)
            {
                if (number > 0)
                {
                    Result[i] = number;
                }
                else
                {
                    Result[i] = Convert.ToSingle((rand.NextDouble() * 2 - 1));
                }
            }
            random.Dispose();
            random1.Dispose();
            random2.Dispose();
            return Result;
        }

        public static double[,] GenerateMatrix(int x, int y, int number, bool increment)
        {
            double[,] Result = new double[x, y];

            for (int i = 0; i < x; ++i)
            {
                for (int j = 0; j < y; j++)
                {
                    if (!increment)
                    {
                        Result[i, j] = number--;
                    }
                    else
                    {
                        Result[i, j] = number++;
                    }
                }
            }
            return Result;
        }

        public static double[,] GenerateMatrix(int x, int y, int number)
        {
            double[,] Result = new double[x, y];

            for (int i = 0; i < x; ++i)
            {
                for (int j = 0; j < y; j++)
                {
                    Result[i, j] = number;
                }
            }
            return Result;
        }

        public static double[,] GenerateMatrix(int x, int y, int seed, double min = 0.1, double max = 0.9)
        {
            Random rand = new Random(seed);
            double[,] Result = new double[x, y];

            for (int i = 0; i < x; ++i)
            {
                for (int j = 0; j < y; j++)
                {
                    Result[i, j] = rand.NextDouble() * max - min;
                }
            }
            return Result;
        }

        public static double[,] GenerateMatrix(int x, int y, double min = 0.1, double max = 0.9, bool incrementInt = false, bool setNumber = false, double number = 0)
        {
            /// super randomizer
            /// järkyttävä overkill mutta olkoot

            RNGCryptoServiceProvider random = new RNGCryptoServiceProvider();
            var Bytes = new byte[4];
            random.GetBytes(Bytes);

            Random rand = new Random(BitConverter.ToInt32(Bytes, 0));

            double[,] Result = new double[x, y];

            int count = 0;

            for (int i = 0; i < x; ++i)
            {
                for (int j = 0; j < y; j++)
                {
                    if (incrementInt)
                    {
                        Result[i, j] = count;
                        count++;
                    }
                    else
                    {
                        Result[i, j] = (setNumber ? number : Convert.ToSingle((rand.NextDouble() * max) - (min)));
                    }
                }
            }
            random.Dispose();

            return Result;
        }

        public static double[,,] InitKernels(int x, int y, int z, double min = 0.1, double max = 0.9)
        {
            /// super randomizer
            /// järkyttävä overkill mutta olkoot

            RNGCryptoServiceProvider random = new RNGCryptoServiceProvider();
            var Bytes = new byte[4];
            random.GetBytes(Bytes);

            Random rand = new Random(BitConverter.ToInt32(Bytes, 0));

            var Result = new double[x, y, z];

            for (int k = 0; k < z; k++)
            {
                for (int i = 0; i < x; i++)
                {
                    for (int j = 0; j < y; j++)
                    {
                        Result[i, j, k] = rand.NextDouble() * max - min;
                    }
                }
            }
            random.Dispose();
            return Result;
        }

        public static double[] LabelVectorCreator(int Size, int Pos)
        {
            var array = new double[Size];
            Array.Clear(array, 0, array.Length);
            array[Pos] = 1;
            return array;
        }

        public static double[] ApplyConnectionMask(int[] mask, double[] input)
        {
            var newInput = new double[mask.Length];
            for (int j = 0; j < mask.Length; j++)
            {
                newInput[j] = input[mask[j]];
            }
            return newInput;
        }

        public static void ResizeImageFolders(string[] path, int x, int y)
        {
            var newRootDir = "";
            var newSubDir = "";

            if (!Directory.Exists(Directory.GetParent(path[0]).FullName + "\\Resized"))
            {
                newRootDir = Directory.GetParent(path[0]).FullName + "\\Resized";
                Directory.CreateDirectory(newRootDir);
                DirectoryInfo dInfo = new DirectoryInfo(newRootDir);
                DirectorySecurity dSecurity = dInfo.GetAccessControl();
                dSecurity.AddAccessRule(new FileSystemAccessRule(new SecurityIdentifier(WellKnownSidType.WorldSid, null), FileSystemRights.FullControl, InheritanceFlags.ObjectInherit | InheritanceFlags.ContainerInherit, PropagationFlags.NoPropagateInherit, AccessControlType.Allow));
                dInfo.SetAccessControl(dSecurity);
            }
            else
            {
                Directory.Delete(Directory.GetParent(path[0]).FullName + "\\Resized", true);
                newRootDir = Directory.GetParent(path[0]).FullName + "\\Resized";
                Directory.CreateDirectory(newRootDir);
                DirectoryInfo dInfo = new DirectoryInfo(newRootDir);
                DirectorySecurity dSecurity = dInfo.GetAccessControl();
                dSecurity.AddAccessRule(new FileSystemAccessRule(new SecurityIdentifier(WellKnownSidType.WorldSid, null), FileSystemRights.FullControl, InheritanceFlags.ObjectInherit | InheritanceFlags.ContainerInherit, PropagationFlags.NoPropagateInherit, AccessControlType.Allow));
                dInfo.SetAccessControl(dSecurity);
            }

            for (int i = 0; i < path.Length; i++)
            {
                if (!Directory.Exists(newRootDir + "\\" + Path.GetFileName(path[i])))
                {
                    newSubDir = newRootDir + "\\" + Path.GetFileName(path[i]);
                    Directory.CreateDirectory(newSubDir);
                    /*
                    DirectoryInfo dInfo = new DirectoryInfo(newSubDir);
                    DirectorySecurity dSecurity = dInfo.GetAccessControl();
                    dSecurity.AddAccessRule(new FileSystemAccessRule(new SecurityIdentifier(WellKnownSidType.WorldSid, null), FileSystemRights.FullControl, InheritanceFlags.ObjectInherit | InheritanceFlags.ContainerInherit, PropagationFlags.NoPropagateInherit, AccessControlType.Allow));
                    dInfo.SetAccessControl(dSecurity);
                    */
                }

                var labels = Directory.GetDirectories(path[i]);

                for (int j = 0; j < labels.Length; j++)
                {
                    var newFilePath = "";
                    if (!Directory.Exists(newSubDir + "\\" + Path.GetFileName(labels[j])))
                    {
                        newSubDir = newRootDir + "\\" + Path.GetFileName(path[i]);
                        newFilePath = newSubDir + "\\" + Path.GetFileName(labels[j]);
                        Directory.CreateDirectory(newFilePath);
                        /*
                        DirectoryInfo dInfo = new DirectoryInfo(newFilePath);
                        DirectorySecurity dSecurity = dInfo.GetAccessControl();
                        dSecurity.AddAccessRule(new FileSystemAccessRule(new SecurityIdentifier(WellKnownSidType.WorldSid, null), FileSystemRights.FullControl, InheritanceFlags.ObjectInherit | InheritanceFlags.ContainerInherit, PropagationFlags.NoPropagateInherit, AccessControlType.Allow));
                        dInfo.SetAccessControl(dSecurity);
                        */
                    }

                    // Console.WriteLine(i);
                    string[] files = Directory.GetFiles(labels[j]);

                    Parallel.ForEach(files, (file) =>
                    {
                        Bitmap bm = (Bitmap)Image.FromFile(file);
                        var fname = Path.GetFileName(file);
                        var img = new Bitmap(bm, new Size(x, y));

                        Console.WriteLine(fname);
                        var newPath = newFilePath + "\\" + fname;
                        img.Save(newPath);

                        bm.Dispose();
                        img.Dispose();
                    });
                }
            }
        }

        public static double[] ImageToArray(string path)
        {
            Bitmap img = (Bitmap)Image.FromFile(path);

            var Result = new double[img.Height * img.Width];

            for (int i = 0; i <= img.Height - 1; i++)
            {
                for (int j = 0; j <= img.Width - 1; j++)
                {
                    Color pixel = img.GetPixel(j, i);
                    double color = (pixel.R + pixel.B + pixel.G) / 3;

                    //Result[img.Height * i + j] = color;
                    Result[img.Height * i + j] = color / 255;
                }
            }

            img.Dispose();

            return Result;
        }

        public static void DrawFromArray(double[,] img)
        {
            var len = Math.Sqrt(img.Length);

            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < len; j++)
                {
                    Console.Write(img[i, j]);
                }
                Console.WriteLine();
            }
        }

        public static void DrawFromArray(double[] img)
        {
            var len = (int)Math.Sqrt(img.Length);

            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < len; j++)
                {
                    Console.Write(img[(i * len) + j]);
                }
                Console.WriteLine();
            }
        }
    }
}
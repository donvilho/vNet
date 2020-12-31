using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace vNet.Dll
{
    public static class cpp
    {
        public const string CppFuncs = @"C:\Users\Viert\Desktop\harkka\vNet\x64\Debug\CppFuncs.dll";

        [DllImport(CppFuncs, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Addf(int a, int b);
    }
}
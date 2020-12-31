#define MyFuncs _declspec(dllexport)

extern "C" {
	MyFuncs int Addf(int a, int b) {
		return a + b;
	}
}
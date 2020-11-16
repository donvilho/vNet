namespace vNet
{
    internal class Result
    {
        public int Position;
        public float ErrorRate;
        public bool Correct;

        public Result()
        {
            ErrorRate = 0;
        }
    }
}
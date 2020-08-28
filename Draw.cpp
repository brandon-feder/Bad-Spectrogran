class Draw
{
    public:
        static void graghData(unsigned int *pixels, unsigned int *dft)
        {
            for(int row = 0; row < WIDTH-1; ++row)
            {
                for(int col = 0; col < HEIGHT; col++)
                {
                    pixels[ XY_ToInt( row, col ) ] = pixels[ XY_ToInt( row+1, col ) ];
                }
            }


            for(int col = 0; col < HEIGHT; col++)
            {
                int value = dft[((FRAME_SIZE/2) * col)/HEIGHT ];
                float percentage = (float)value / ((float)FRAME_SIZE/100 );
                std::cout << percentage << "\n";
                pixels[ XY_ToInt( WIDTH-1, col ) ] = lerpColor(
                    rgbToInt(0, 0, 255),
                    rgbToInt(255, 0, 0),
                    ( percentage > 1 ) ? 1 : percentage
                );
            }
        }
};

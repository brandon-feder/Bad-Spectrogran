
// Converts rgba to an integer
int rgbToInt(int r, int g, int b)
{
    int rgb = r;
    rgb = (rgb << 8) + g;
    rgb = (rgb << 8) + b;
    rgb = (rgb << 8) + 0;
    return rgb;
}

// Lerp Color
int lerpColor(int c1, int c2, float t)
{
    return t * ( c1 + c2) + c1;
}
// Gets the index in an array representing pixels
int XY_ToInt(int x, int y)
{
    return x + y * WIDTH;
}

// Gets a range of audio from the full raw buffer
void getAudioFrame(unsigned int *frame, float time, int duration, unsigned int *buffer, int bufferSize)
{

    int center =  bufferSize * time/duration;

    for(int i = 0; i < FRAME_SIZE; ++i)
    {
        if( (center + i - FRAME_SIZE/2 < 0) || (center + i - FRAME_SIZE/2 < 0) > bufferSize)
        {
            frame[i] = 0;
        } else
        {
            frame[i] = buffer[center + i - FRAME_SIZE/2];
        }
    }
}

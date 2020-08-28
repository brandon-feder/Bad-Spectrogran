const int WIDTH = 1500;
const int HEIGHT = 1000;

const int FRAME_SIZE = 5000; // Must Be Less Than # Of Possble GPU Threads

// Number of cores per block
const int nCoresPerBlock = 1024;

// Number of cores per block that will be used
const int N_CORES = FRAME_SIZE/2 % (nCoresPerBlock+1);

// Number Of Blocks
const int N_BLOCKS = FRAME_SIZE/2 / N_CORES + 1;

// Whether to print the FPS
const bool PRINT_FPS = true;

// The path to the wav file to play
const char *AUDIO_PATH = "./media/sin_test.wav";

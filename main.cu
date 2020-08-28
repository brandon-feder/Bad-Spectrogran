#include <SDL2/SDL.h>
#include <SDL2/SDL_audio.h>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <math.h>

#include "Settings.cpp"
#include "DFT.cu"
#include "helpers.cpp"
#include "Draw.cpp"

std::chrono::steady_clock::time_point start;

SDL_AudioSpec wavSpec;
Uint32 wavLength;
Uint8 *wavBuffer;
SDL_AudioDeviceID deviceId;
float duration;
unsigned int *rawData;

SDL_Window * window;
SDL_Renderer * renderer;
SDL_Texture * tex;

void initSDL();
void playAudio();
void updateFrame(unsigned int *pixels);
void cleanUpSDL();

int main(int argc, char ** argv)
{
    // Initialize SDL and Related Variables
    initSDL();

    unsigned int* pixels;
    cudaMallocManaged( &pixels, WIDTH*HEIGHT*sizeof( int ) );

    // Start Playing the Audio
    playAudio();

    cudaSetDevice(0);
    // Get the start time of the audio
    start = std::chrono::steady_clock::now();

    bool quit = false; // Flag for when to quit the program
    while (!quit)
    {
        // Get the start time of the current frame
        std::chrono::steady_clock::time_point frameStart = std::chrono::steady_clock::now();

        // Calculate the time since the audio started playing
        float time = std::chrono::duration_cast<std::chrono::microseconds>(frameStart - start).count()/1000000.0;

        // Test whether the file is done playing or not
        if( time >= duration )
        {
            quit = true;
        } else // If not done playing
        {
            // Calculate the next frame of data
            unsigned int *frame;
            cudaMallocManaged( &frame, FRAME_SIZE*sizeof( int ) );
            getAudioFrame(frame, time, duration, rawData, wavLength/wavSpec.channels );
            
            // Setup the array containing the DFT data
            unsigned int *dft = new unsigned int[ FRAME_SIZE/2 ];
            cudaMallocManaged( &dft, FRAME_SIZE/2*sizeof( int ) );

            // Call the gpu func to calculate the DFT
            DFT<<<N_BLOCKS, N_CORES>>>(frame, dft);
            cudaError_t error = cudaDeviceSynchronize();

            if( error != 0 )
            {
                std::cout << "Erro on GPU: " << error << "\n";
            }


            Draw::graghData(pixels, dft);

            // Handle close event
            SDL_Event event;
            while (SDL_PollEvent(&event)) { // For every event in queue
                switch (event.type) // If event is a quit event
                {
                    case SDL_QUIT:
                        quit = true; // quit
                        break;
                }
            }

            // Update the SDL texture with all the data
            updateFrame(pixels);

            // Print the FPS if PRINT_FPS Flag is True
            if( PRINT_FPS )
            {
                std::chrono::steady_clock::time_point frameEnd = std::chrono::steady_clock::now();
                int duration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart).count();
                std::cout << "FPS: " << 1000000/(float)duration << "\n";
            }

            // Clean Up
            cudaFree( dft );
            cudaFree( frame );
        }
    }

    // Clean Up
    cleanUpSDL();
    cudaDeviceReset();
    cudaFree( pixels );

    delete[] rawData;

    return 0;
}


void initSDL()
{
    // Initialize SDL
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Init(SDL_INIT_AUDIO);

    // Creat SDL Window
    window = SDL_CreateWindow("Real Time DFT", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);

    // Setip the render and texture
    renderer = SDL_CreateRenderer(window, -1, 0);
    tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);

    SDL_LoadWAV(AUDIO_PATH, &wavSpec, &wavBuffer, &wavLength);
    deviceId = SDL_OpenAudioDevice(NULL, 0, &wavSpec, NULL, 0);
    SDL_QueueAudio(deviceId, wavBuffer, wavLength);

    duration = (float)wavLength / ( wavSpec.freq * wavSpec.channels * SDL_AUDIO_BITSIZE(wavSpec.format)/8  );

    rawData = new unsigned int[ wavLength/wavSpec.channels ];

    for(int i = 1; i < wavLength; i += 2)
    {
        if(i % 2 == 1)
        {
            if( wavBuffer[i] > 256/2 )
            {
                rawData[(i-1)/2] = 256 - wavBuffer[i]  + 256/2;
            } else
            {
                rawData[(i-1)/2] = 256 - wavBuffer[i] - 256/2;
            }
        }
    }
}

void playAudio()
{
    SDL_PauseAudioDevice(deviceId, 0);
}

void updateFrame(unsigned int *pixels)
{
    SDL_UpdateTexture(tex, NULL, pixels, WIDTH * sizeof(Uint32));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, tex, NULL, NULL);
    SDL_RenderPresent(renderer);
}

void cleanUpSDL()
{
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    SDL_CloseAudioDevice(deviceId);
    SDL_FreeWAV(wavBuffer);

    SDL_Quit();
}

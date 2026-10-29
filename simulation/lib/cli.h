#ifndef CLI_H
#define CLI_H

typedef struct {
    float windSpeed;
    int visualizationMode;
    int collisionMode;
    int renderDuration;
    char outputPath[256];
    char modelPath[512];
    float reynoldsNumber;
    int scaleFromCLI;
    int gridX, gridY, gridZ;
    float smagorinskyCs;
    int useMRT;
    int useEntropic;
    float turbulenceIntensity;
    int spongeCells; // -1 = auto (sizeX/8), 0 = off
    char vtkOutputPath[256];
    int vtkInterval;
    int vtkStats;
} CliOptions;

// Parse command-line options. Returns 0 on success, 1 if --help was
// requested (caller should exit cleanly).
int cli_parse(int argc, char *argv[], CliOptions *opts);

#endif // CLI_H

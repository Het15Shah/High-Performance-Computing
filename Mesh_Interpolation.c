#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

// Define structure for points
typedef struct {
    double x, y;
} Points;

// Global variables for grid and points
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, NTHR_C, Maxiter;
double dx, dy;  // Grid spacing

// Read scattered points from binary file
void readPoints(FILE *file, Points *points) {
    size_t result = fread(points, sizeof(Points), NUM_Points, file);
    if (result != NUM_Points) {
        printf("Error: Unable to read all points from the file. Expected %d, but read %zu\n", NUM_Points, result);
        exit(1);
    }
}

// Write mesh data to output file
void printmesh(double *meshValue) {
    FILE *fd = fopen("Mesh.out", "w");
    if (fd == NULL) {
        printf("Error: Unable to create output file\n");
        exit(1);
    }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", meshValue[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}

// Optimized Parallel Interpolation with OpenMP Reduction
void interpolation(double *meshValue, Points *points) {
    memset(meshValue, 0, GRID_X * GRID_Y * sizeof(double));

    // Allocate thread-private mesh arrays
    double *privateMesh = (double *)calloc(GRID_X * GRID_Y * NTHR_C, sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *localMesh = &privateMesh[tid * GRID_X * GRID_Y];

        #pragma omp for schedule(dynamic, 1000) nowait
        for (int i = 0; i < NUM_Points; i++) {
            double px = points[i].x, py = points[i].y;
            double weighing = 1.0;

            int gx = (int)(px / dx);
            int gy = (int)(py / dy);
            if (gx < 0 || gx >= NX-1 || gy < 0 || gy >= NY-1) continue;

            double lx = px - gx * dx;
            double ly = py - gy * dy;
            
            int p1 = gy * GRID_X + gx;
            int p2 = gy * GRID_X + (gx + 1);
            int p3 = (gy + 1) * GRID_X + gx;
            int p4 = (gy + 1) * GRID_X + (gx + 1);

            double a1 = (dx - lx) * (dy - ly);
            double a2 = lx * (dy - ly);
            double a3 = (dx - lx) * ly;
            double a4 = lx * ly;

            localMesh[p1] += a1 * weighing;
            localMesh[p2] += a2 * weighing;
            localMesh[p3] += a3 * weighing;
            localMesh[p4] += a4 * weighing;
        }

        // Merge private results into global mesh with OpenMP reduction
        #pragma omp for
        for (int j = 0; j < GRID_X * GRID_Y; j++) {
            for (int k = 0; k < NTHR_C; k++) {
                meshValue[j] += privateMesh[k * GRID_X * GRID_Y + j];
            }
        }
    }

    free(privateMesh);
}

// Log execution time
void log_execution_time(const char *input_name, int num_threads, double exec_time) {
    FILE *logfile = fopen("new.csv", "a");
    if (logfile == NULL) {
        printf("Error: Unable to open execution_times.csv\n");
        return;
    }
    fprintf(logfile, "%s,%d,%lf\n", input_name, num_threads, exec_time);
    fclose(logfile);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <num_threads>\n", argv[0]);
        return 1;
    }

    char filename[50];
    strcpy(filename, argv[1]);
    NTHR_C = atoi(argv[2]);
    omp_set_num_threads(NTHR_C);

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        return 1;
    }

    // Read grid dimensions
    fread(&NX, sizeof(int), 1, file);
    fread(&NY, sizeof(int), 1, file);
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter, sizeof(int), 1, file);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // Allocate memory for mesh values and points
    double *meshValue = (double*) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points*) calloc(NUM_Points, sizeof(Points));

    double elapsed_time = 0.0;

    for (int iteration = 0; iteration < Maxiter; iteration++) {
        readPoints(file, points);
        double start_time = omp_get_wtime();
        interpolation(meshValue, points);
        elapsed_time += omp_get_wtime() - start_time;
    }

    printmesh(meshValue);
    printf("Interpolation execution time = %lf seconds\n", elapsed_time);

    // Extract filename without extension
    char *base_name = strtok(filename, ".");
    
    // Log execution time
    log_execution_time(base_name, NTHR_C, elapsed_time);

    free(meshValue);
    free(points);
    fclose(file);
    return 0;
}

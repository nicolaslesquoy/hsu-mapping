#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Test functions
#define THRESHOLD 1e-9f

// Define the grid structure

typedef struct point_t{
    float x; // X coordinate
    float y; // Y coordinate (set to 0 if 1D)
    float z; // Z coordinate (set to 0 if 1D or 2D)
    int value_dim; // Number of dimensions for the value (scalar = 1, vector = 2 or 3)
    float value[3]; // Value associated with the point (can be a scalar or vector)
} point_t;

typedef struct grid_t{
    int dim[3]; // Dimensions of the grid. 1D = {1, 0, 0}, 2D = {1, 1, 0}, 3D = {1, 1, 1}
    size_t size; // Total number of elements in the grid
    point_t *points; // Pointer to the array of points in the grid
} grid_t;

// Read from a CSV file and populate the grid structure
grid_t read_csv(const char *filename, int grid_dim, int value_dim) {
    grid_t grid = {0};
    if (grid_dim < 1 || grid_dim > 3) {
        fprintf(stderr, "Error: grid_dim must be 1, 2, or 3\n");
        exit(EXIT_FAILURE);
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Initialize grid dimensions
    grid.dim[0] = 1;
    grid.dim[1] = (grid_dim >= 2) ? 1 : 0;
    grid.dim[2] = (grid_dim == 3) ? 1 : 0;
    grid.size = 0;
    grid.points = NULL;
    char line[256];

    // Count the number of points (skip header)
    while (fgets(line, sizeof(line), file)) {
        grid.size++;
    }
    if (grid.size <= 1) {
        fprintf(stderr, "Error: No data rows found in %s\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    grid.size--; // Exclude header
    rewind(file);

    // Allocate memory for points
    grid.points = (point_t *)malloc(grid.size * sizeof(point_t));
    if (!grid.points) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Skip the header line
    fgets(line, sizeof(line), file);

    for (size_t i = 0; i < grid.size; i++) {
        if (!fgets(line, sizeof(line), file)) {
            fprintf(stderr, "Error: Unexpected end of file at line %zu\n", i + 2);
            free(grid.points);
            fclose(file);
            exit(EXIT_FAILURE);
        }
        point_t *point = &grid.points[i];
        point->value_dim = value_dim;

        // Parse the line (cases unchanged)
        if (grid_dim == 1 && value_dim == 1) {
            sscanf(line, "%f,%f", &point->x, &point->value[0]);
            point->y = 0.0f; point->z = 0.0f;
            point->value[1] = 0.0f; point->value[2] = 0.0f;
        } else if (grid_dim == 2 && value_dim == 1) {
            sscanf(line, "%f,%f,%f", &point->x, &point->y, &point->value[0]);
            point->z = 0.0f;
            point->value[1] = 0.0f; point->value[2] = 0.0f;
        } else if (grid_dim == 3 && value_dim == 1) {
            sscanf(line, "%f,%f,%f,%f", &point->x, &point->y, &point->z, &point->value[0]);
            point->value[1] = 0.0f; point->value[2] = 0.0f;
        } else if (grid_dim == 3 && value_dim == 3) {
            sscanf(line, "%f,%f,%f,%f,%f,%f", &point->x, &point->y, &point->z,
                   &point->value[0], &point->value[1], &point->value[2]);
        } else {
            fprintf(stderr, "Unsupported grid_dim (%d) or value_dim (%d)\n", grid_dim, value_dim);
            free(grid.points);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
    return grid;
}

void print_grid(const grid_t *grid) {
    if (!grid || !grid->points) {
        printf("Grid is empty or not initialized.\n");
        return;
    }
    for (size_t i = 0; i < grid->size; i++) {
        const point_t *point = &grid->points[i];
        printf("Point %zu: (%.8f, %.8f, %.8f) -> Value: (", i, point->x, point->y, point->z);
        for (int j = 0; j < point->value_dim; j++) {
            printf("%.8f", point->value[j]);
            if (j < point->value_dim - 1) printf(", ");
        }
        printf(")\n");
    }
}

// RBF interpolation functions

float multiquadric(float r, float shape_parameter) {
    return sqrtf(r * r + shape_parameter * shape_parameter);
}

float inverse_multiquadric(float r, float shape_parameter) {
    return 1.0f / multiquadric_r(r, shape_parameter);
}

float gaussian(float r, float support_radius, float shape_parameter) {
    if (isfinite(support_radius)) {
        // Compact support
        float threshold = sqrtf(-logf(THRESHOLD) / shape_parameter);
        float _support_radius = fminf(support_radius, threshold);
        float _deltaY = expf(-((shape_parameter * _support_radius) * (shape_parameter * _support_radius)));
        if (r > _support_radius) {
            return 0.0f;
        } else {
            return expf(-((shape_parameter * r) * (shape_parameter * r))) + _deltaY;
        }
    } else {
        // Global support
        return expf(-((shape_parameter * r) * (shape_parameter * r)));
    }
}

// Weight finding function

typedef struct {
    float *weights; // Pointer to the weights array
    int size;       // Number of weights
} weights_t;

weights_t find_weights(const grid_t *grid, const float shape_parameter, const float support_radius) {
    weights.size = grid->size;
    if (weights.size <= 0) {
        fprintf(stderr, "Error: Grid is empty, cannot find weights.\n");
        return weights;
    }
    weights.weights = (float *)malloc(weights.size * sizeof(float));
    if (!weights.weights) {
        fprintf(stderr, "Error: Memory allocation for weights failed.\n");
        exit(EXIT_FAILURE);
    }
    // Create pairwise matrix 
    

}
    

int main(void) {
    float x[2] = {0.1f, 0.2f};
    float center[2] = {0.5f, 0.5f};
    float mq = multiquadric(x, center, 2, 0.1f);
    float imq = inverse_multiquadric(x, center, 2, 0.1f);
    float gauss = gaussian(x, center, 2, INFINITY, 0.1f);
    printf("Multiquadric: %.8f\n", mq);
    printf("Inverse Multiquadric: %.8f\n", imq);
    printf("Gaussian: %.8f\n", gauss);
    return 0;
}

// int main(int argc, char *argv[]) {
//     if (argc < 4) {
//         fprintf(stderr, "Usage: %s <filename> <grid_dim> <value_dim>\n", argv[0]);
//         return EXIT_FAILURE;
//     }

//     const char *filename = argv[1];
//     int grid_dim = atoi(argv[2]);
//     int value_dim = atoi(argv[3]);

//     grid_t grid = read_csv(filename, grid_dim, value_dim);
//     print_grid(&grid);

//     // Free allocated memory
//     free(grid.points);
//     return EXIT_SUCCESS;
// }
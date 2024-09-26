Wala' Essam Ashqar
12027854

#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define SIZE 256

float __attribute__((aligned(16))) vec1[SIZE];
float __attribute__((aligned(16))) vec2[SIZE];
float __attribute__((aligned(16))) mat1[SIZE][SIZE];
float __attribute__((aligned(16))) mat2[SIZE][SIZE];

double seconds(){
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1000000000.0;}

float vector(){
    float prod = 0;
    for (int i = 0; i < SIZE; i++){
        prod += vec1[i] * vec2[i];}
    return prod;}

float vector_sse(){
    float prod = 0;
    __m128 X, Y, Z;
    Z[0] = Z[1] = Z[2] = Z[3] = 0;
    for (int i = 0; i < SIZE; i += 4){
        X = _mm_load_ps(&vec1[i]);
        Y = _mm_load_ps(&vec2[i]);
        X = _mm_mul_ps(X, Y);
        Z = _mm_add_ps(X, Z);}
    prod = Z[0] + Z[1] + Z[2] + Z[3];
    return prod;}

float mat_vector(){
    float res[SIZE] = { 0 };
    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            res[i] += mat1[i][j] * vec1[j];}}
    float sum = 0;
    for (int i = 0; i < SIZE; i++){
        sum += res[i];}
    return sum;}

float mat_vector_sse(){
    float res[SIZE];
    __m128 X, Y, Z;

    for (int i = 0; i < SIZE; i++){
        Z[0] = Z[1] = Z[2] = Z[3] = 0;
        for (int j = 0; j < SIZE; j += 4){
            X = _mm_load_ps(&mat1[i][j]);
            Y = _mm_load_ps(&vec1[j]);
            X = _mm_mul_ps(X, Y);
            Z = _mm_add_ps(X, Z);}
        res[i] = Z[0] + Z[1] + Z[2] + Z[3];
    }
    float sum = 0;
    for (int i = 0; i < SIZE; i++){
        sum += res[i];}
    return sum;
}

float mat(){
    float res[SIZE][SIZE] = { 0 };
    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            for (int k = 0; k < SIZE; k++){
                res[i][j] += mat1[i][k] * mat2[k][j];}}}
    float sum = 0;
    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            sum += res[i][j];}}
    return sum;}

float mat_sse(){
    float res[SIZE][SIZE];
    __m128 X, Y, Z;
    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j += 4){
            Z[0] = Z[1] = Z[2] = Z[3] = 0;
            for (int k = 0; k < SIZE; k++){
                X = _mm_set1_ps(mat1[i][k]);
                Y = _mm_loadu_ps(&mat2[k][j]);
                X = _mm_mul_ps(X, Y);
                Z = _mm_add_ps(X, Z);}
            _mm_store_ps(&res[i][j], Z);}}
    float sum = 0;
    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            sum += res[i][j];}}
    return sum;}

void execute(const char* label, float (*function)()) {
    double start, end;
    start = seconds();
    float result = function();
    end = seconds();
    printf("%s Time: %lf\n", label, end - start);}


int main(){
    double start, end;
    float vres;
    float v_sse_res, mat_v_res, mat_v_sse_res;
    float mat_res, mat_sse_res;

    for (int i = 0; i < SIZE; i++){
        vec1[i] = rand() % 2;
        vec2[i] = rand() % 2;}
    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            mat1[i][j] = rand() % 2;
            mat2[i][j] = rand() % 2;}}

    execute("(1) V - V", &vector);
    execute("(2) V - V", &vector_sse);
    execute("(1) M - V", &mat_vector);
    execute("(2) M - V", &mat_vector_sse);
    execute("(1) M - M", &mat);
    execute("(2) M - M", &mat_sse);
    return 0;}
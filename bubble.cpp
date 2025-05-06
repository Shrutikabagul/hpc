#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        // Odd-even transposition sort logic
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> original(n);
    cout << "Enter elements: ";
    for (int& x : original) {
        cin >> x;
    }

    // Sequential Bubble Sort
    vector<int> arr1 = original;
    double start = omp_get_wtime();
    sequentialBubbleSort(arr1);
    double end = omp_get_wtime();
    cout << "\nSequential Bubble Sort Result:\n";
    for (int x : arr1) cout << x << " ";
    cout << "\nTime: " << (end - start) << " seconds\n";

    // Parallel Bubble Sort
    vector<int> arr2 = original;
    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();
    cout << "\nParallel Bubble Sort Result:\n";
    for (int x : arr2) cout << x << " ";
    cout << "\nTime: " << (end - start) << " seconds\n";

    return 0;
}

/*g++ -fopenmp bubble.cpp -o bubble
./bubble */

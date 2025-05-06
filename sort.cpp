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

// Corrected Parallel Bubble Sort (Odd-Even Transposition Sort)
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        int start = i % 2;
        #pragma omp parallel for
        for (int j = start; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Merge function
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    while (i <= mid && j <= right) {
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int m = 0; m < k; m++) arr[left + m] = temp[m];
}

// Sequential Merge Sort
void sequentialMergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    sequentialMergeSort(arr, left, mid);
    sequentialMergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    #pragma omp parallel sections
    {
        #pragma omp section
        parallelMergeSort(arr, left, mid);

        #pragma omp section
        parallelMergeSort(arr, mid + 1, right);
    }
    merge(arr, left, mid, right);
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> input(n);
    cout << "Enter elements: ";
    for (int i = 0; i < n; i++) cin >> input[i];

    vector<int> bubbleSeq = input, bubblePar = input;
    vector<int> mergeSeq = input, mergePar = input;

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    sequentialBubbleSort(bubbleSeq);
    end = omp_get_wtime();
    cout << "\nSequential Bubble Sort Time: " << (end - start) << " sec" << endl;

    // Parallel Bubble Sort
    start = omp_get_wtime();
    parallelBubbleSort(bubblePar);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << (end - start) << " sec" << endl;

    // Sequential Merge Sort
    start = omp_get_wtime();
    sequentialMergeSort(mergeSeq, 0, n - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << (end - start) << " sec" << endl;

    // Parallel Merge Sort
    start = omp_get_wtime();
    parallelMergeSort(mergePar, 0, n - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << (end - start) << " sec" << endl;

    // Output sorted arrays
    cout << "\nSorted Array (Bubble Sort): ";
    for (int x : bubblePar) cout << x << " ";
    cout << "\nSorted Array (Merge Sort): ";
    for (int x : mergePar) cout << x << " ";
    cout << endl;

    return 0;
}


/*g++ -fopenmp sort.cpp -o sort
./sort
*/

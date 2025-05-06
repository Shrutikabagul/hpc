#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Merge Function
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right)
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
    while (i <= mid)
        temp[k++] = arr[i++];
    while (j <= right)
        temp[k++] = arr[j++];

    for (int m = 0; m < k; ++m)
        arr[left + m] = temp[m];
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

    vector<int> arr(n), arr2;
    cout << "Enter elements: ";
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
    }
    arr2 = arr;

    // Sequential
    double start = omp_get_wtime();
    sequentialMergeSort(arr, 0, n - 1);
    double end = omp_get_wtime();
    cout << "Sequential Merge Sort Result:\n";
    for (int x : arr) cout << x << " ";
    cout << "\nTime: " << (end - start) << " seconds\n";

    // Parallel
    start = omp_get_wtime();
    parallelMergeSort(arr2, 0, n - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Result:\n";
    for (int x : arr2) cout << x << " ";
    cout << "\nTime: " << (end - start) << " seconds\n";

    return 0;
}

/*g++ -fopenmp merge.cpp -o merge
./merge */

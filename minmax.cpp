#include <iostream>
#include <omp.h>
#include <limits>
using namespace std;

int main() {
    int N;
    cout << "Enter number of elements: ";
    cin >> N;

    int *arr = new int[N];
    cout << "Enter elements:\n";
    for (int i = 0; i < N; i++)
        cin >> arr[i];

    int sum = 0, min_val = INT_MAX, max_val = INT_MIN;

    #pragma omp parallel for reduction(+:sum) reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < N; i++) {
        sum += arr[i];
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
    }

    float average = static_cast<float>(sum) / N;

    cout << "Sum = " << sum << endl;
    cout << "Min = " << min_val << endl;
    cout << "Max = " << max_val << endl;
    cout << "Average = " << average << endl;

    delete[] arr;
    return 0;
}
/*g++ -fopenmp minmax.cpp -o minmax
./minmax
*/

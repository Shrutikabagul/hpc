// dfs_parallel.cpp
#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

void parallelDFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    stack<int> st;

    st.push(start);
    visited[start] = true;

    cout << "Parallel DFS traversal: ";

    while (!st.empty()) {
        int u;

        #pragma omp critical
        {
            u = st.top();
            st.pop();
        }

        cout << u << " ";

        #pragma omp parallel for
        for (int i = 0; i < graph[u].size(); ++i) {
            int v = graph[u][i];
            if (!visited[v]) {
                #pragma omp critical
                {
                    if (!visited[v]) {
                        visited[v] = true;
                        st.push(v);
                    }
                }
            }
        }
    }
    cout << endl;
}

int main() {
    int n, e, u, v;
    cout << "Enter number of nodes and edges: ";
    cin >> n >> e;

    vector<vector<int>> graph(n);

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < e; ++i) {
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // undirected
    }

    int start;
    cout << "Enter start node: ";
    cin >> start;

    parallelDFS(graph, start);
    return 0;
}
/*g++ -fopenmp dfs.cpp -o dfs
./parallel_dfs
Enter number of nodes and edges: 3 3
Enter edges (u v):
0 1
0 2
1 2
Enter start node: 0
Parallel DFS traversal: 0 1 2
*/


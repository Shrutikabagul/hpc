// bfs_parallel.cpp
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

void parallelBFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    cout << "Parallel BFS traversal: ";

    while (!q.empty()) {
        int size = q.size();
        vector<int> currentLevel;

        while (size--) {
            int u = q.front();
            q.pop();
            cout << u << " ";
            currentLevel.push_back(u);
        }

        #pragma omp parallel for
        for (int i = 0; i < currentLevel.size(); ++i) {
            int u = currentLevel[i];
            for (int v : graph[u]) {
                if (!visited[v]) {
                    #pragma omp critical
                    {
                        if (!visited[v]) {
                            visited[v] = true;
                            q.push(v);
                        }
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

    parallelBFS(graph, start);
    return 0;
}
/*g++ -fopenmp bfs.cpp -o bfs
./bfs
Enter number of nodes and edges: 3 3
Enter edges (u v):
0 1
0 2
1 2
Enter start node: 0
Parallel BFS traversal: 0 1 2
*/

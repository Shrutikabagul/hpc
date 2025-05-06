#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

void parallel_bfs(const vector<vector<int>>& adj, int start, vector<bool>& visited) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    cout << "BFS starting from node " << start << ": ";

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << node << " ";

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); ++i) {
            int neighbor = adj[node][i];
            if (!visited[neighbor]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }
    cout << endl;
}

void parallel_dfs(const vector<vector<int>>& adj, int start, vector<bool>& visited) {
    stack<int> s;
    s.push(start);
    visited[start] = true;

    cout << "DFS starting from node " << start << ": ";

    while (!s.empty()) {
        int node = s.top();
        s.pop();
        cout << node << " ";

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); ++i) {
            int neighbor = adj[node][i];
            if (!visited[neighbor]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        s.push(neighbor);
                    }
                }
            }
        }
    }
    cout << endl;
}

int main() {
    int n, e;
    cout << "Enter number of vertices and edges: ";
    cin >> n >> e;

    vector<vector<int>> adj(n);
    cout << "Enter " << e << " edges (u v):\n";
    for (int i = 0; i < e; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }

    vector<bool> visited_bfs(n, false);
    vector<bool> visited_dfs(n, false);

    parallel_bfs(adj, 0, visited_bfs);
    parallel_dfs(adj, 0, visited_dfs);

    return 0;
}


/*g++ -fopenmp bfs_dfs.cpp -o bfs_dfs
./bfs_dfs
Enter number of vertices and edges: 5 4
Enter 4 edges (u v):
0 1
0 2
1 3
3 4
BFS starting from node 0: 0 1 2 3 4
DFS starting from node 0: 0 1 3 4 2
*/

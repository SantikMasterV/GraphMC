#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include "pugixml.hpp"
#include <utility>

using namespace std;
namespace fs = std::filesystem;

struct Edge {
    int src;
    int tgt;
    int offset;
};

/*
string get_attribute(const string& line, const string& key) {
    size_t key_pos = line.find(key);
    if (key_pos == string::npos) return "";

    size_t quote_start = line.find("\"", key_pos);
    if (quote_start == string::npos) return "";

    size_t quote_end = line.find("\"", quote_start + 1);
    if (quote_end == string::npos) return "";

    return line.substr(quote_start + 1, quote_end - quote_start - 1);
}

map<int, int> parse_graph_from_xml(const string& file_path, int num_cells, vector<pair<int, int>>& edges_out) {
    ifstream file(file_path);
    string line;

    map<int, int> base_vertices;
    vector<Edge> base_edges;
    int index = 0;

    while (getline(file, line)) {
        // Очистка від пробілів на початку/кінці
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // VERTEX
        if (line.find("<VERTEX") != string::npos) {
            size_t id_start = line.find("id = \"");
            if (id_start != string::npos) {
                id_start += 6;
                size_t id_end = line.find("\"", id_start);
                if (id_end != string::npos) {
                    string id_str = line.substr(id_start, id_end - id_start);
                    try {
                        int id = stoi(id_str);
                        base_vertices[id] = index++;
                    }
                    catch (...) {
                        cerr << "Invalid VERTEX id: \"" << id_str << "\"" << endl;
                    }
                }
            }
        }

        // EDGE
        else if (line.find("<EDGE") != string::npos) {
            int src = -1, tgt = -1, offset = 0;

            for (int i = 0; i < 2; ++i) {
                if (!getline(file, line)) break;
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);

                if (line.find("<SOURCE") != string::npos) {
                    size_t v_start = line.find("vertex = \"");
                    if (v_start != string::npos) {
                        v_start += 10;
                        size_t v_end = line.find("\"", v_start);
                        if (v_end != string::npos) {
                            string v_str = line.substr(v_start, v_end - v_start);
                            try {
                                src = stoi(v_str);
                            }
                            catch (...) {
                                cerr << "Invalid SOURCE vertex: \"" << v_str << "\"" << endl;
                            }
                        }
                    }
                }

                else if (line.find("<TARGET") != string::npos) {
                    size_t v_start = line.find("vertex = \"");
                    if (v_start != string::npos) {
                        v_start += 10;
                        size_t v_end = line.find("\"", v_start);
                        if (v_end != string::npos) {
                            string v_str = line.substr(v_start, v_end - v_start);
                            try {
                                tgt = stoi(v_str);
                            }
                            catch (...) {
                                cerr << "Invalid TARGET vertex: \"" << v_str << "\"" << endl;
                            }
                        }
                    }

                    size_t o_start = line.find("offset = \"");
                    if (o_start != string::npos) {
                        o_start += 10;
                        size_t o_end = line.find("\"", o_start);
                        if (o_end != string::npos) {
                            string o_str = line.substr(o_start, o_end - o_start);
                            try {
                                offset = stoi(o_str);
                            }
                            catch (...) {
                                cerr << "Invalid offset: \"" << o_str << "\"" << endl;
                            }
                        }
                    }
                }
            }

            if (src != -1 && tgt != -1) {
                base_edges.push_back({ src, tgt, offset });
            }
        }
    }

    // Розгортання графа
    int base_size = base_vertices.size();
    map<int, int> vertices;

    for (int i = 0; i < num_cells * base_size; ++i)
        vertices[i] = 0;

    for (int cell = 0; cell < num_cells; ++cell) {
        int offset = cell * base_size;
        for (const auto& e : base_edges) {
            if (base_vertices.count(e.src) && base_vertices.count(e.tgt)) {
                int new_src = offset + base_vertices[e.src];
                int new_tgt = offset + base_vertices[e.tgt] + e.offset * base_size;
                if (new_tgt < num_cells * base_size)
                    edges_out.emplace_back(new_src, new_tgt);
            }
        }
    }

    return vertices;
}*/

/*
pair<map<int, int>, vector<pair<int, int>>> parse_graph_from_xml(const string& file_path, int num_cells) {
    map<int, int> base_vertices;  // Mapping from XML vertex ID to new 0-based index
    vector<tuple<int, int, int>> base_edges;  // (src, tgt, offset)

    pugi::xml_document doc;
    if (!doc.load_file(file_path.c_str())) {
        cerr << "Error: Could not load XML file " << file_path << endl;
        return { {}, {} };
    }

    // Read vertices from XML and create a mapping
    int index = 0;
    for (pugi::xml_node vertex : doc.child("GRAPH").children("VERTEX")) {
        int xml_id = vertex.attribute("id").as_int();
        base_vertices[xml_id] = index++;
    }

    // Read edges from XML
    for (pugi::xml_node edge : doc.child("GRAPH").children("EDGE")) {
        pugi::xml_node source = edge.child("SOURCE");
        pugi::xml_node target = edge.child("TARGET");

        int source_id = source.attribute("vertex").as_int();
        int target_id = target.attribute("vertex").as_int();

        string offset_str = target.attribute("offset").as_string();
        int offset = offset_str.empty() ? 0 : stoi(offset_str);

        // Convert to mapped indices
        if (base_vertices.count(source_id) && base_vertices.count(target_id)) {
            int src = base_vertices[source_id];
            int tgt = base_vertices[target_id];
            base_edges.emplace_back(src, tgt, offset);
        }
    }

    // Construct the expanded graph
    int num_base_vertices = base_vertices.size();
    map<int, int> vertices;
    vector<pair<int, int>> edges;

    for (int i = 0; i < num_cells; i++) {
        int cell_offset = i * num_base_vertices;  // Offset for the current cell

        for (const auto& [src, tgt, offset] : base_edges) {
            int new_src = cell_offset + src;
            int new_tgt = cell_offset + tgt + (offset * num_base_vertices);

            if (new_tgt < num_cells * num_base_vertices) {  // Ensure within valid range
                edges.emplace_back(new_src, new_tgt);
            }
        }
    }

    // Initialize vertices (assuming spin=1 for all)
    for (int i = 0; i < num_cells * num_base_vertices; i++) {
        vertices[i] = 1;
    }

    return { vertices, edges };
}*/
string get_attribute_value(const string& line, const string& key) {
    size_t pos = line.find(key);
    if (pos == string::npos) return "";

    pos = line.find('=', pos);
    if (pos == string::npos) return "";

    // Пропустити пробіли
    while (pos < line.size() && isspace(line[pos + 1])) ++pos;

    if (line[pos + 1] == '"' || line[pos + 1] == '\'') {
        char quote = line[pos + 1];
        size_t start = pos + 2;
        size_t end = line.find(quote, start);
        if (end != string::npos)
            return line.substr(start, end - start);
    }

    return "";
}

pair<map<int, int>, vector<pair<int, int>>> parse_graph_from_xml(const string& file_path, int num_cells) {
    map<int, int> base_vertices;
    vector<tuple<int, int, int>> base_edges;

    // Читаємо файл
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << file_path << endl;
        return {};
    }

    string line;
    vector<string> lines;
    while (getline(file, line)) {
        lines.push_back(line);
    }

    // Обробка вершин
    int vertex_index = 0;
    for (const auto& l : lines) {
        if (l.find("<VERTEX") != string::npos) {
            string id_str = get_attribute_value(l, "id");
            if (!id_str.empty()) {
                int xml_id = stoi(id_str);
                base_vertices[xml_id] = vertex_index++;
            }
        }
    }

    // Обробка ребер
    for (const auto& l : lines) {
        if (l.find("<EDGE") != string::npos) {
            int source_id = -1, target_id = -1, offset = 0;

            string src_str = get_attribute_value(l, "vertex");  // Поверне перший vertex
            size_t src_pos = l.find("<SOURCE");
            if (src_pos != string::npos) {
                src_str = get_attribute_value(l.substr(src_pos), "vertex");
            }

            size_t tgt_pos = l.find("<TARGET");
            if (tgt_pos != string::npos) {
                string tgt_str = get_attribute_value(l.substr(tgt_pos), "vertex");
                string offset_str = get_attribute_value(l.substr(tgt_pos), "offset");

                if (!tgt_str.empty()) target_id = stoi(tgt_str);
                if (!offset_str.empty()) offset = stoi(offset_str);
            }

            if (!src_str.empty()) source_id = stoi(src_str);

            if (base_vertices.count(source_id) && base_vertices.count(target_id)) {
                int src = base_vertices[source_id];
                int tgt = base_vertices[target_id];
                base_edges.emplace_back(src, tgt, offset);
            }
        }
    }

    // Побудова повного графа
    int num_base_vertices = static_cast<int>(base_vertices.size());
    map<int, int> vertices;
    for (int i = 0; i < num_cells * num_base_vertices; ++i) {
        vertices[i] = 1;
    }

    vector<pair<int, int>> edges;
    for (int i = 0; i < num_cells; ++i) {
        int cell_offset = i * num_base_vertices;
        for (const auto& [src, tgt, offset] : base_edges) {
            int new_src = cell_offset + src;
            int new_tgt = cell_offset + tgt + (offset * num_base_vertices);
            if (new_tgt < num_cells * num_base_vertices) {
                edges.emplace_back(new_src, new_tgt);
            }
        }
    }

    return { vertices, edges };
}

void initialize_spins(map<int, int>& vertices) {
    for (auto& [id, spin] : vertices)
        spin = (rand() % 2) == 0 ? -1 : 1;
}

double local_energy_change(const map<int, int>& vertices, const vector<pair<int, int>>& edges, double J, double H, int v) {
    int sum_neighbors = 0;
    for (auto [v1, v2] : edges) {
        if (v1 == v)
            sum_neighbors += vertices.at(v2);
        else if (v2 == v)
            sum_neighbors += vertices.at(v1);
    }
    return 2 * vertices.at(v) * (H + J * sum_neighbors);
}

void metropolis_step(map<int, int>& vertices, const vector<pair<int, int>>& edges, double J, double H, double beta) {
    int v = rand() % vertices.size();
    auto it = vertices.begin();
    advance(it, v);
    int site = it->first;

    double dE = local_energy_change(vertices, edges, J, H, site);
    if (dE < 0 || (rand() / (double)RAND_MAX) < exp(-beta * dE)) {
        vertices[site] *= -1;
    }
}
double simulate(map<int, int>& vertices, const vector<pair<int, int>>& edges, double J, double H, double beta, int steps, int loops) {
    vector<double> magnetizations;

    for (int i = 0; i < steps * loops; ++i) {
        metropolis_step(vertices, edges, J, H, beta);
        if (i % steps == 0 && i > 0) {
            double m = 0.0;
            for (auto& [id, s] : vertices)
                m += s;
            magnetizations.push_back(m / vertices.size());
        }
    }

    double avg_m = 0;
    for (double m : magnetizations) avg_m += m;
    return avg_m / magnetizations.size();
}

int main() {
    srand(time(0));
    int length = 250;
    double J = -1.0;
    double beta = 5.0;
    int steps = 100000;
    int loops = 12;

    vector<double> fields;
    for (int i = 0; i <= 40; ++i)
        fields.push_back(-4.0 + i * 0.2);

    for (int i = 9; i <= 9; ++i) {
        string folder = "data/" + to_string(i);
        string path = folder + "/graph1.xml";
        if (!fs::exists(path)) {
            cout << "File " << path << " not found. Skipping.\n";
            continue;
        }

        cout << "Processing " << path << "...\n";

        // Используем новую функцию parse_graph_from_xml
        auto [vertices, edges] = parse_graph_from_xml(path, length);

        // Вывод информации о графе
        cout << "  Found " << vertices.size() << " vertices and " << edges.size() << " edges\n";

        // Вывод первых 10 рёбер для проверки
        cout << "  First 10 edges: [";
        for (size_t i = 0; i < min(edges.size(), static_cast<size_t>(10)); ++i) {
            if (i != 0) cout << ", ";
            cout << "(" << edges[i].first << ", " << edges[i].second << ")";
        }
        cout << "]" << endl;

        ofstream out(folder + "/result.txt");
        out << "M H\n";

        for (double H : fields) {
            initialize_spins(vertices);
            double M = simulate(vertices, edges, J, H, beta, steps, loops);
            out << fixed << setprecision(6) << M << " " << H << "\n";
        }
    }

    return 0;
}
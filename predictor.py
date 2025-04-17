import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import matplotlib.pyplot as plt
from torch.functional import F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# === Парсинг XML-графа ===
def parse_unitcell(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    node_map = {}
    edge_index = []

    for i, vertex in enumerate(root.findall('VERTEX')):
        vid = int(vertex.attrib['id'])
        node_map[vid] = i  # перенумерація вершин

    # Фіча для кожної вершини – поки просто одиниця (1)
    x = torch.ones((len(node_map), 1), dtype=torch.float)

    for edge in root.findall('EDGE'):
        source = int(edge.find('SOURCE').attrib['vertex'])
        target = int(edge.find('TARGET').attrib['vertex'])
        i = node_map[source]
        j = node_map[target]
        edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def check_result_lengths(root_dir, expected_len=41):
    for i in range(1, 1000):  # перевіримо перші 1000 папок (або скільки треба)
        subdir = os.path.join(root_dir, str(i))
        result_path = os.path.join(subdir, 'result.txt')

        if not os.path.isfile(result_path):
            continue

        try:
            df = pd.read_csv(result_path, delim_whitespace=True, header=0)
            actual_len = len(df)
            if actual_len != expected_len:
                print(f"[!] {result_path} має {actual_len} рядків (очікується {expected_len})")
        except Exception as e:
            print(f"[X] Не вдалося зчитати {result_path}: {e}")


# Використай це:

def parse_graph_from_xml(file_path, num_cells):
    tree = ET.parse(file_path)
    root = tree.getroot()

    base_vertices = {}  # Mapping from XML vertex ID to new 0-based index
    base_edges = []

    # Read vertices from XML and create a mapping
    for index, vertex in enumerate(root.findall('VERTEX')):
        xml_id = int(vertex.get('id'))  # Original ID from XML
        base_vertices[xml_id] = index  # Map to 0-based index

    # Read edges from XML
    for edge in root.findall('EDGE'):
        source_id = int(edge.find('SOURCE').get('vertex'))
        target_id = int(edge.find('TARGET').get('vertex'))

        offset_str = edge.find('TARGET').get('offset')
        offset = int(offset_str) if offset_str is not None else 0

        # Convert to mapped indices
        if source_id in base_vertices and target_id in base_vertices:
            src = base_vertices[source_id]
            tgt = base_vertices[target_id]
            base_edges.append((src, tgt, offset))

    # Construct the expanded graph
    num_base_vertices = len(base_vertices)  # Corrected count of unique vertices
    vertices = {i: 1 for i in range(num_cells * num_base_vertices)}
    edges = []

    for i in range(num_cells):
        cell_offset = i * num_base_vertices  # Offset for the current cell

        for src, tgt, offset in base_edges:
            new_src = cell_offset + src
            new_tgt = cell_offset + tgt + (offset * num_base_vertices)  # Apply offset from XML

            if new_tgt < num_cells * num_base_vertices:  # Ensure within valid range
                edges.append((new_src, new_tgt))

    return vertices, edges

def parse_replicated_unitcell(xml_path, n_rep=20, prt=False ):
    vertices, edges = parse_graph_from_xml(xml_path,  n_rep)

    num_total_vertices = len(vertices)
    num_vertices_per_cell = num_total_vertices // n_rep
    degree_dict = {i: 0 for i in range(num_total_vertices)}
    for src, tgt in edges:
        degree_dict[src] += 1
        degree_dict[tgt] += 1  # якщо граф неорієнтований

    # Ознака: ступінь вершини
    degree_feat = torch.tensor([degree_dict[i] for i in range(num_total_vertices)], dtype=torch.float).unsqueeze(1)

    # Ознака: локальний індекс в комірці (0, 1, ..., N-1)
    cell_index_feat = torch.tensor(
        [i % num_vertices_per_cell for i in range(num_total_vertices)], dtype=torch.float
    ).unsqueeze(1)

    # Комбінуємо всі фічі в один тензор [num_nodes, num_features]
    x = torch.cat([degree_feat, cell_index_feat], dim=1)
    #print(vertices.keys())
    #x = torch.ones((len(vertices),1), dtype=torch.float)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if (prt):
        print(edges)
        print(edge_index)
    return Data(x=x, edge_index=edge_index)

def load_magnetization(magnetization_path):
    df = pd.read_csv(magnetization_path, delim_whitespace=True, header=0)
    return torch.tensor(df['M'].values, dtype=torch.float32)
    # Файл result.txt повинен містити дві колонки: M та H, розділені пробілом.
    '''df = pd.read_csv(magnetization_path, comment='#', delim_whitespace=True, header=None, names=['M', 'H'])
    df = df.sort_values('H')
    # Ми передбачаємо, що H зафіксовані для всіх графів і хочемо передбачити всю криву M(H)
    # Повертаємо тензор форми [41]
    return torch.tensor(df['M'].values, dtype=torch.float)'''

def load_field(magnetization_path):
    df = pd.read_csv(magnetization_path, comment='#', delim_whitespace=True, header=0, names=['M', 'H'])
    #df = df.sort_values('H')
    return df['H'].values
# === Клас датасету ===
class MagnetizationDataset(Dataset):
    def __init__(self, root_dir, n_rep= 100):
        super(MagnetizationDataset, self).__init__()
        self.subdirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.n_rep = n_rep

    def len(self):
        return len(self.subdirs)

    def get(self, idx):
        subdir = self.subdirs[idx]
        graph_path = os.path.join(subdir, 'graph1.xml')
        result_path = os.path.join(subdir, 'result.txt')
        graph = parse_replicated_unitcell(graph_path, self.n_rep)
        # Додаємо додаткову розмірність – тепер y має форму [1, 41]
        graph.y = load_magnetization(result_path).unsqueeze(0)
        return graph


def add_features_to_graph(data, cell_index):
    # Ступінь вершини
    from torch_geometric.utils import degree
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).unsqueeze(1)

        # Номер комірки для всіх вершин
    cell_feat = torch.full((data.num_nodes, 1), float(cell_index))

        # Додаємо (concatenate) ступінь і номер комірки
    data.x = torch.cat([deg, cell_feat], dim=1)


    return data
# === Модель ===
class GCNMagnetModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GCNMagnetModel, self).__init__()
        #self.initial_conv = GCNConv(2, output_dim)
        self.conv1 = GCNConv(2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.pool = global_mean_pool
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.out = nn.Linear(hidden_dim * 2, 41)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, 41),
            nn.Tanh()
        )
    def forward(self, x, edge_index, batch):
        # Два шари GCN
        #hidden = self.initial_conv(x, edge_index)
        #hidden = F.tanh(hidden)

        #hidden = torch.relu(self.conv1(x, edge_index))
        hidden = self.conv1(x, edge_index)
        hidden = F.tanh(hidden)
        #hidden = torch.relu(self.conv2(hidden , edge_index))
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden) #

        hidden = torch.cat([gmp(hidden, batch),
                            gap(hidden, batch)], dim=1)

        out = self.out(hidden)
        return out
        '''x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        # Глобальне усереднення по вузлах графа
        x = self.pool(x, batch)
        x = torch.relu(self.fc1(x))

        return self.fc2(x)'''


# === Навчання моделі ===
def train_model():
    # Дані зберігаються у папках Data/1, Data/2, ...
    dataset = MagnetizationDataset('Data')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Визначаємо розмір вихідного вектора як кількість значень M для одного графа,
    # наприклад, для першого графа – tensor з формою [41]
    sample_y = load_magnetization(os.path.join(dataset.subdirs[0], 'result.txt'))
    output_dim = len(sample_y)

    model = GCNMagnetModel(hidden_dim=256, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    model.train()
    for epoch in range(100):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            # batch.y зараз має форму [batch_size, 1, output_dim]
            # Перетворимо його до [batch_size, output_dim]
            loss = criterion(out, batch.y.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

        torch.save(model.state_dict(), 'mag_model.pt')
    return model


# === Предсказание для нового графа ===
def predict_magnetization(model, graph_path):
    model.eval()
    data = parse_unitcell(graph_path)
    for i in range(1, 51):
        data = add_features_to_graph(data, cell_index=1)
    # Для одного графа всі вузли належать до однієї партії (batch index = 0)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    with torch.no_grad():
        prediction = model(data.x, data.edge_index, data.batch)
    # Повертаємо передбачену криву (тензор з формою [output_dim])
    return prediction.squeeze().numpy()


# === Основний запуск програми ===
if __name__ == '__main__':
    check_result_lengths('Data', expected_len=41)
    model = train_model()
    # Приклад використання для нового графа:
    graph_path = os.path.join('Data', '22', 'graph1.xml')
    result_path = os.path.join('Data', '22', 'result.txt')
    parse_replicated_unitcell(graph_path, n_rep=20, prt=True)
    # Отримуємо передбачення намагнічення
    predicted_M = predict_magnetization(model, graph_path)
    # Отримуємо вектор H (з них будуємо графік)
    H_values = load_field(result_path)

    # Побудова графіка
    plt.figure(figsize=(8, 5))
    plt.plot(H_values, predicted_M, marker='o', label='Передбачене M(H)', color='b')

    # Якщо хочеш порівняти з реальними даними
    real_M = load_magnetization(result_path).numpy()
    plt.plot(H_values, real_M, marker='x', label='Реальне M(H)', color='r')

    plt.xlabel('H (зовнішнє поле)')
    plt.ylabel('M (намагнічення)')
    plt.title('Крива намагнічення')
    plt.legend()
    plt.grid(True)
    plt.show()

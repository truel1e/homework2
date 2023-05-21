#Данное задание было выполнено совместными усилиями целиком, усердно, упорно, с энтузиазмом :) Все функции и классы делали вместе. Большой вклад в первую подзадачу внес Нелюбин Д.А., во вторую - Прокофьев А.В., третью подзадачу мы делали совместными усилиями.
import sys, random, time
import matplotlib.pyplot as plt

class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.E = init_graph
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        '''
        Этот метод обеспечивает симметричность графика. Другими словами, если существует путь от узла A к B со значением V, должен быть путь от узла B к узлу A со значением V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value

        return graph

    def get_nodes(self):
        "Возвращает узлы графа"
        return self.nodes

    def get_vertexes(self):
        "Возвращает список смежности графа"
        return self.E

    def get_outgoing_edges(self, node):
        "Возвращает соседей узла"
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2):
        "Возвращает значение ребра между двумя узлами."
        return self.graph[node1][node2]


def Dijkstra_slow(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    # Мы будем использовать этот словарь, чтобы сэкономить на посещении каждого узла и обновлять его по мере продвижения по графику
    shortest_path = {}

    # Мы будем использовать этот dict, чтобы сохранить кратчайший известный путь к найденному узлу
    previous_nodes = {}

    # Мы будем использовать max_value для инициализации значения "бесконечности" непосещенных узлов
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # Однако мы инициализируем значение начального узла 0
    shortest_path[start_node] = 0

    # Алгоритм выполняется до тех пор, пока мы не посетим все узлы
    while unvisited_nodes:
        # Приведенный ниже блок кода находит узел с наименьшей оценкой
        current_min_node = None
        for node in unvisited_nodes:
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        # Приведенный ниже блок кода извлекает соседей текущего узла и обновляет их расстояния
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                previous_nodes[neighbor] = current_min_node

        # После посещения его соседей мы отмечаем узел как "посещенный"
        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path


def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node

    while node != start_node:
        path.append(node)
        try:
            node = previous_nodes[node]
        except:
            print("Найден следующий лучший маршрут с ценностью -1 в вершину" + ' ' + target_node) #Нет пути в вершину v
            return
    # Добавить начальный узел вручную
    path.append(start_node)
    print("Найден следующий лучший маршрут с ценностью {} в вершину".format(shortest_path[target_node]) + ' ' + target_node)
    print(" -> ".join(reversed(path)))

def graph_read(data):
    text = []  # Необработанный список элементов из заданного графа в 'dijkstraData.txt'
    with open(data,'r') as file:  # Читаем файл 'dijkstraData.txt' в котором содержится наш граф и заполняем text
        for item in file:
            text.append(item)
    text2 = []  # Обработанный список элементов из заданного графа в 'dijkstraData.txt', где каждый первый элемент внутри каждого списка - сама вершина, а остальные элементы - ребра.
    for i in text: #Заполняем text2
        text2.append(list(i.split()))
    nodes = [] #Создаем список вершин графа
    len_text = len(text2)
    for i in range(1, len_text + 1): #Заполняем наш список вершин (каждая вершина - строка, т.к. наш алгоритм Дейкстры работает со строками)
     nodes.append(str(i))
    init_graph = {} #Множество ребер, где ключ - это вершина, а значение - множество, внутри которого ключ - другая вершина, смежная с первой, а значение - вес между ними.
    for node in nodes:
     init_graph[node] = {}
    edges4node = [] #Массив, где внутри массивов мы храним ребра для вершин.
    for i in text2:
        edges = []
        for j in i[1:]:
         edges.append(j)
        edges4node.append(edges)
    n = 0
    for i in edges4node: #Заполяем наше множество init_graph. n - вершина, edge - другая вершина, size - вес между ними.
     n += 1
     for j in i:
        edge, size = j.split(',') #Разъединяем число с запятой на две части, левая - вершина, правая - вес.
        init_graph[str(n)][edge] = int(size)

    graph = Graph(nodes, init_graph) #инициализируем граф
    previous_nodes, shortest_path = Dijkstra_slow(graph=graph, start_node="1")
    for i in nodes: #Печатаем результат
        print_result(previous_nodes, shortest_path, start_node="1", target_node=i)

print('Пример работы алгоритмы Дейкстра на заданном в подзадаче 1 графе:')
graph_read('dijkstraData.txt')
#Отсюда начинается подзадача 2
print('')
print('Пример алгоритма Dijkstra_slow на случайно сгенерированном графе(начинаем с вершины 0):')
def generate_graph(n, p): #Функция, которая генерирует случайный неориентированный взвешенный граф
    nodes = [] #Список вершин
    init_graph = {} #Множество вершин и ребер к ним прилегающим
    N = n + 1
    for i in range(N): #Заполняем список вершин
        nodes.append(str(i))
    for j in nodes: #Заполняем множество ребер
        init_graph[str(j)] = {}
    m = -1
    for i in range(N): #С вероятностью p добавляем ребро в множество init_graph со случайным весом от 1 до 1000)
        m += 1
        for j in range(i+1, N):
            if random.uniform(0,1) < p:
                init_graph[str(m)][str(j)] = random.randint(1,1000)
    g = Graph(nodes,init_graph) #создаем граф и возвращаем его
    return g
#Снизу - пример работы Dijkstra_slow на случайно сгенерированном неориентированном взвешенном графе
nodes = []
for i in range(101):
    nodes.append(str(i))

g = generate_graph(100, 0.5)
previous_nodes, shortest_path = Dijkstra_slow(g, start_node="0")
for i in nodes:  # Печатаем результат
    print_result(previous_nodes, shortest_path, start_node="0", target_node=i)

#Отсюда начинается подзадача 3
class BinaryHeap: #Класс двоичной кучи
    def __init__(self):
        self.heap = []

    def push(self, value):
        self.heap.append(value)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        elif len(self.heap) == 1:
            return self.heap.pop()
        else:
            root = self.heap[0]
            self.heap[0] = self.heap.pop()
            self._sift_down(0)
            return root

    def _sift_up(self, index):
        parent_index = (index - 1) // 2
        if index > 0 and self.heap[parent_index] > self.heap[index]:
            self.heap[parent_index], self.heap[index] = self.heap[index], self.heap[parent_index]
            self._sift_up(parent_index)

    def _sift_down(self, index):
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        min_index = index
        if left_child_index < len(self.heap) and self.heap[left_child_index] < self.heap[min_index]:
            min_index = left_child_index
        if right_child_index < len(self.heap) and self.heap[right_child_index] < self.heap[min_index]:
            min_index = right_child_index
        if min_index != index:
            self.heap[index], self.heap[min_index] = self.heap[min_index], self.heap[index]
            self._sift_down(min_index)

def Dijkstra_fast(graph, start): #Алгоритм Дейсктры с использованием двоичной кучи
    heap = BinaryHeap() #Инициализируем нашу двоичную кучу
    vertexes = graph.get_vertexes() #Возьмем списки смежности из нашего графа
    distances = {vertex: float('inf') for vertex in vertexes} #Изначально инициализируем расстояния как infinity
    distances[start] = 0
    heap.push((0, start)) #Инициализируем значение начального узла 0
    while heap.heap: #Сам алгоритм Дейсктры с участием двоичной кучи
        current_distance, current_vertex = heap.pop()
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in vertexes[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heap.push((distance, neighbor))
    return distances

#Пример работы Dijkstra_fast на случайно сгенерированном графе(начинаем с вершины 0):
distances = Dijkstra_fast(g, '0')
print('')
print('Пример работы Dijkstra_fast на случайно сгенерированном графе(начинаем с вершины 0):')
print(distances)

#Ниже представлены графики для Dijkstra_slow, Dijkstra_fast для случайно сгенерированного графа(начиная с вершины 0) (работает только в ноутбуке)
n = 1000
A = 100
B = 1000
step = 100
x = []
y1 = []
y2 = []
g = generate_graph(100, 0.5)
for i in range(A, B+1, step):
    x.append(i)
    t1 = time.time()
    Dijkstra_slow(g, '0')
    t2 = time.time()
    Dijkstra_fast(g, '0')
    t3 = time.time()
    y1.append(t2 - t1)
    y2.append(t3 - t2)

plt.plot(x, y1, label='Dijkstra_slow')
plt.plot(x, y2, label='Dijkstra_fast')
plt.xlabel('Number of vertices')
plt.ylabel('Time (s)')
plt.legend()
plt.show()
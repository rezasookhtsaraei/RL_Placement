import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import time
import copy
import scipy.stats as stats
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PRIORITY_WEIGHTS = {
    1: {  
        'response_time': 0.6,
        'load_balance': 0.1,
        'reliability': 0.3
    },
    2: { 
        'response_time': 0.5,
        'load_balance': 0.2,
        'reliability': 0.3
    },
    3: {  
        'response_time': 0.3,
        'load_balance': 0.6,
        'reliability': 0.1
    }
}

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)

class DiabetesDAGEnv:
    def __init__(self, num_dags=50):
        self.num_dags = num_dags
        self.num_tasks = 8  

        self.num_nodes = 52  
        self.node_hierarchy = self._create_hierarchical_structure()

        self.delay_matrix, self.bandwidth_matrix = self._create_network_matrices()

        self.dags = []
        self.dag_priorities = []  
        self.dag_deadlines = []   
        self.dag_arrival_times = []  
        self.dag_completion_times = []  
        self.dag_makespans = []  

        arrival_rate = 0.5 
        for i in range(num_dags):
            if i == 0:
                arrival_time = 0.0 
            else:
                arrival_time = self.dag_arrival_times[-1] + np.random.exponential(1.0/arrival_rate)

            dag, priority, deadline = self._generate_random_dag_with_priority_and_deadline()
            self.dags.append(dag)
            self.dag_priorities.append(priority)
            self.dag_deadlines.append(deadline)
            self.dag_arrival_times.append(arrival_time)

        self.original_arrival_times = self.dag_arrival_times.copy()

        self.current_dag_idx = 0
        self.last_completed_dag_info = None  
        self.debug_backup_creation = False  
        self.episode_allocation_history = []  
        self.episode_timing_history = [] 
        self.deadline_violations_by_priority = {1: 0, 2: 0, 3: 0}  
        self.reset()

    def _create_hierarchical_structure(self):
        hierarchy = {
            'levels': 4,  
            'node_layers': {},  
            'ram_capacity': np.zeros(self.num_nodes, dtype=np.float32),  
            'storage_capacity': np.zeros(self.num_nodes, dtype=np.float32), 
            'computing_power': np.zeros(self.num_nodes, dtype=np.float32),  
            'reliability': np.zeros(self.num_nodes, dtype=np.float32),  
            'node_type': {}  
        }

        hierarchy['computing_power'][0] = np.float32(300000 / 100000)  
        hierarchy['ram_capacity'][0] = np.float32(128.0)  
        hierarchy['storage_capacity'][0] = np.float32(5000.0) 
        hierarchy['reliability'][0] = np.float32(np.random.uniform(0.0, 1.0))  
        hierarchy['node_layers'][0] = 0
        hierarchy['node_type'][0] = 'cloud'

        for i in range(1, 26):
            hierarchy['computing_power'][i] = np.float32(80000 / 100000)  
            hierarchy['ram_capacity'][i] = np.float32(16.0)  
            hierarchy['storage_capacity'][i] = np.float32(500.0)  
            hierarchy['reliability'][i] = np.float32(np.random.uniform(0.0, 1.0))  
            hierarchy['node_layers'][i] = 1
            hierarchy['node_type'][i] = 'fog2'

        for i in range(26, 51):
            hierarchy['computing_power'][i] = np.float32(20000 / 100000)  
            hierarchy['ram_capacity'][i] = np.float32(4.0)  
            hierarchy['storage_capacity'][i] = np.float32(64.0)  
            hierarchy['reliability'][i] = np.float32(np.random.uniform(0.0, 1.0))  
            hierarchy['node_layers'][i] = 2
            hierarchy['node_type'][i] = 'fog1'

        hierarchy['computing_power'][51] = np.float32(4000 / 100000) 
        hierarchy['ram_capacity'][51] = np.float32(0.5)  
        hierarchy['storage_capacity'][51] = np.float32(8.0)  
        hierarchy['reliability'][51] = np.float32(np.random.uniform(0.0, 1.0))  
        hierarchy['node_layers'][51] = 3
        hierarchy['node_type'][51] = 'edge'

        return hierarchy

    def _create_network_matrices(self):
        delay_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        bandwidth_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    delay_matrix[i, j] = 0.01 
                    bandwidth_matrix[i, j] = 1000.0  
                    type_i = self.node_hierarchy['node_type'][i]
                    type_j = self.node_hierarchy['node_type'][j]

                    if type_i == 'cloud' and type_j == 'fog2':
                        bandwidth = np.random.uniform(500.0, 1000.0)
                        delay = np.random.uniform(20.0, 80.0) / 1000.0  
                    elif type_i == 'fog2' and type_j == 'cloud':
                        bandwidth = np.random.uniform(500.0, 1000.0)
                        delay = np.random.uniform(20.0, 80.0) / 1000.0

                    elif type_i == 'fog2' and type_j == 'fog1':
                        bandwidth = 500.0
                        delay = np.random.uniform(8.0, 15.0) / 1000.0
                    elif type_i == 'fog1' and type_j == 'fog2':
                        bandwidth = 500.0
                        delay = np.random.uniform(8.0, 15.0) / 1000.0

                    elif type_i == 'fog1' and type_j == 'edge':
                        bandwidth = 100.0
                        delay = np.random.uniform(2.0, 5.0) / 1000.0
                    elif type_i == 'edge' and type_j == 'fog1':
                        bandwidth = 100.0
                        delay = np.random.uniform(2.0, 5.0) / 1000.0

                    elif type_i == 'cloud' and type_j == 'fog1':
                        bandwidth = 250.0  
                        delay = np.random.uniform(30.0, 60.0) / 1000.0
                    elif type_i == 'fog1' and type_j == 'cloud':
                        bandwidth = 250.0
                        delay = np.random.uniform(30.0, 60.0) / 1000.0

                    elif type_i == 'cloud' and type_j == 'edge':
                        bandwidth = 100.0  
                        delay = np.random.uniform(50.0, 100.0) / 1000.0
                    elif type_i == 'edge' and type_j == 'cloud':
                        bandwidth = 100.0
                        delay = np.random.uniform(50.0, 100.0) / 1000.0

                    elif type_i == 'fog2' and type_j == 'edge':
                        bandwidth = 100.0
                        delay = np.random.uniform(10.0, 20.0) / 1000.0
                    elif type_i == 'edge' and type_j == 'fog2':
                        bandwidth = 100.0
                        delay = np.random.uniform(10.0, 20.0) / 1000.0

                    else:
                        if type_i == type_j:
                            if type_i == 'fog2':
                                bandwidth = np.random.uniform(800.0, 1000.0)
                                delay = np.random.uniform(5.0, 10.0) / 1000.0
                            elif type_i == 'fog1':
                                bandwidth = np.random.uniform(400.0, 600.0)
                                delay = np.random.uniform(3.0, 7.0) / 1000.0
                            else:
                                bandwidth = np.random.uniform(200.0, 300.0)
                                delay = np.random.uniform(1.0, 3.0) / 1000.0
                        else:
                            bandwidth = np.random.uniform(100.0, 200.0)
                            delay = np.random.uniform(10.0, 30.0) / 1000.0

                    delay_matrix[i, j] = delay
                    bandwidth_matrix[i, j] = bandwidth

        return delay_matrix, bandwidth_matrix

    def _generate_random_dag_with_priority_and_deadline(self):
        G = nx.DiGraph()

        priority = np.random.choice([1, 2, 3])

        if priority == 1:  
            deadline = np.random.uniform(0.5, 2.0)  
        elif priority == 2:  
            deadline = np.random.uniform(0.5, 5.0) 
        else:  
            deadline = np.random.uniform(0.5, 10.0)  

        G.add_node(0,
                  name="Compute blood sugar",
                  processing_need=np.float32(300 / 100000), 
                  ram_need=np.float32(32 / 1024),  
                  storage_need=np.float32(5 / 1024)) 

        G.add_node(1,
                  name="Send to doctor",
                  processing_need=np.float32(50 / 100000),  
                  ram_need=np.float32(16 / 1024), 
                  storage_need=np.float32(2 / 1024))  

        G.add_node(2,
                  name="Retrieve patient records",
                  processing_need=np.float32(10000 / 100000),  
                  ram_need=np.float32(64 / 1024),  
                  storage_need=np.float32(20 / 1024)) 

        G.add_node(3,
                  name="Control insulin pump",
                  processing_need=np.float32(2000 / 100000),  
                  ram_need=np.float32(32 / 1024),  
                  storage_need=np.float32(5 / 1024))  

        G.add_node(4,
                  name="Compute insulin level",
                  processing_need=np.float32(3000 / 100000),  
                  ram_need=np.float32(32 / 1024),  
                  storage_need=np.float32(5 / 1024))  

        G.add_node(5,
                  name="Review values",
                  processing_need=np.float32(15000 / 100000),  
                  ram_need=np.float32(96 / 1024),  
                  storage_need=np.float32(15 / 1024))  

        G.add_node(6,
                  name="Compute pump command",
                  processing_need=np.float32(5000 / 100000),  
                  ram_need=np.float32(32 / 1024),  
                  storage_need=np.float32(5 / 1024)) 

        G.add_node(7,
                  name="Log insulin dose",
                  processing_need=np.float32(1000 / 100000),  
                  ram_need=np.float32(16 / 1024),  
                  storage_need=np.float32(2 / 1024))  

        for i in range(8):
            G.nodes[i]['deadline'] = np.float32(np.random.uniform(0.5, 2.0))
            G.nodes[i]['data_size'] = np.float32(np.random.uniform(0.1, 1.0))

        kb_to_gb = 1 / (1024 * 1024)

        G.add_edge(0, 1, data_size=np.float32(2 * kb_to_gb))
        G.add_edge(0, 4, data_size=np.float32(2 * kb_to_gb))
        G.add_edge(1, 2, data_size=np.float32(2 * kb_to_gb))
        G.add_edge(2, 3, data_size=np.float32(20 * kb_to_gb))
        G.add_edge(2, 5, data_size=np.float32(20 * kb_to_gb))
        G.add_edge(4, 5, data_size=np.float32(5 * kb_to_gb))
        G.add_edge(5, 6, data_size=np.float32(50 * kb_to_gb))
        G.add_edge(6, 3, data_size=np.float32(5 * kb_to_gb))
        G.add_edge(6, 7, data_size=np.float32(2 * kb_to_gb))

        assert nx.is_directed_acyclic_graph(G), "Generated graph is not a DAG"

        return G, priority, deadline

    def _calculate_data_transfer_time(self, source_node, target_node, data_size):
        if source_node == target_node:
            return 0.0  

        bandwidth = self.bandwidth_matrix[source_node, target_node]
        return data_size / bandwidth  

    def _get_predecessor_data_transfer_time(self, task_id, current_node):
        max_transfer_time = 0.0
        predecessors = list(self.dag.predecessors(task_id))

        for pred in predecessors:
            if pred in self.task_status['completed']:
                pred_node = self.task_allocations[pred]

                data_size = self.dag[pred][task_id]['data_size']

                transfer_time = self._calculate_data_transfer_time(pred_node, current_node, data_size)
                max_transfer_time = max(max_transfer_time, transfer_time)

        return max_transfer_time

    def _calculate_response_time_paper(self, task_id, node_id):
        
        rdtt = self._get_predecessor_data_transfer_time(task_id, node_id)

        
        queue_load = 0.0
        for queued_task in self.node_queues[node_id][self.current_priority]:
            queue_load += self.dag.nodes[queued_task]['processing_need']

        current_processing_need = self.dag.nodes[task_id]['processing_need']
        computing_power = self.node_hierarchy['computing_power'][node_id]

        pt = (queue_load + current_processing_need) / computing_power

        response_time = rdtt + pt

        return response_time

    def _calculate_execution_time(self, task_id, node_id, priority):
        """محاسبه زمان اجرای یک وظیفه با در نظر گیری اولویت و صف‌های گره"""
        processing_need = self.dag.nodes[task_id]['processing_need']
        computing_power = self.node_hierarchy['computing_power'][node_id]

        base_time = processing_need / computing_power

        queue_load = 0.0
        for p in range(1, priority + 1):
            for task in self.node_queues[node_id][p]:
                queue_load += self.dag.nodes[task]['processing_need']

        execution_time = base_time + (queue_load / computing_power)

        return execution_time

    def _select_backup_node(self, task_id, primary_node, candidate_nodes, priority):
        
        if not candidate_nodes:
            return None

        scores = []

        for node_id in candidate_nodes:
            delay = self.delay_matrix[primary_node, node_id]
            delay_score = 1.0 / (1.0 + delay * 100)  

           
            reliability = self.node_reliabilities[node_id]
            reliability_score = reliability  

           
            total_queue_load = (
                len(self.node_queues[node_id][1]) +
                len(self.node_queues[node_id][2]) +
                len(self.node_queues[node_id][3])
            )

            max_possible_load = self.num_tasks * 3  
            load_score = 1.0 - (total_queue_load / max_possible_load)  

            ram_utilization = self.remaining_ram[node_id] / self.node_ram_capacities[node_id]
            storage_utilization = self.remaining_storage[node_id] / self.node_storage_capacities[node_id]
            capacity_score = (ram_utilization + storage_utilization) / 2.0  

            final_score = delay_score + reliability_score + load_score + capacity_score

            primary_type = self.node_hierarchy['node_type'][primary_node]
            candidate_type = self.node_hierarchy['node_type'][node_id]

            diversity_bonus = 0.1 if primary_type != candidate_type else 0.0

            final_score += diversity_bonus

            scores.append((node_id, final_score, delay, reliability, total_queue_load))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_node = scores[0][0]

        return best_node

    def _create_backup_for_priority_1(self, task_id, primary_node):
        task_ram_need = self.dag.nodes[task_id]['ram_need']
        task_storage_need = self.dag.nodes[task_id]['storage_need']

        candidate_nodes = []
        for node_id in range(self.num_nodes):
            if node_id == primary_node:
                continue  

            if (self.remaining_ram[node_id] >= task_ram_need and
                self.remaining_storage[node_id] >= task_storage_need):

                candidate_nodes.append(node_id)

        if not candidate_nodes:
            return False

        selected_node = self._select_backup_node(task_id, primary_node, candidate_nodes, priority=1)

        if selected_node is None:
            return False

        self.remaining_ram[selected_node] -= task_ram_need
        self.remaining_storage[selected_node] -= task_storage_need

        
        self.node_queues[selected_node][self.current_priority].append(task_id)

       
        backup_execution_time = self._calculate_execution_time(task_id, selected_node, self.current_priority)
        backup_completion_time = self.current_time + backup_execution_time

       
        self.task_status['backup'][task_id] = (selected_node, backup_completion_time)
        self.backup_allocations[task_id] = selected_node  

        return True

    def _create_backup_for_priority_2(self, task_id, primary_node):
       
        avg_reliability = np.mean(self.node_reliabilities)

        task_ram_need = self.dag.nodes[task_id]['ram_need']
        task_storage_need = self.dag.nodes[task_id]['storage_need']

        if self.node_reliabilities[primary_node] < avg_reliability:
            candidate_nodes = []

            for node_id in range(self.num_nodes):
                if node_id == primary_node:
                    continue
                if (self.remaining_ram[node_id] >= task_ram_need and
                    self.remaining_storage[node_id] >= task_storage_need):
                    candidate_nodes.append(node_id)

            if not candidate_nodes:
                return False

            selected_node = self._select_backup_node(task_id, primary_node, candidate_nodes, priority=2)

            if selected_node is None:
                return False

            self.remaining_ram[selected_node] -= task_ram_need
            self.remaining_storage[selected_node] -= task_storage_need
            self.node_queues[selected_node][self.current_priority].append(task_id)

            backup_execution_time = self._calculate_execution_time(task_id, selected_node, self.current_priority)
            backup_completion_time = self.current_time + backup_execution_time

            self.task_status['backup'][task_id] = (selected_node, backup_completion_time)
            self.backup_allocations[task_id] = selected_node  

            return True

        return False

    def _calculate_load_deviation(self):
        
        cpu_demands = np.zeros(self.num_nodes, dtype=np.float32)
        ram_demands = np.zeros(self.num_nodes, dtype=np.float32)

        for node_id in range(self.num_nodes):
            for priority in [1, 2, 3]:
                for task_id in self.node_queues[node_id][priority]:
                    cpu_demands[node_id] += self.dag.nodes[task_id]['processing_need']
                    ram_demands[node_id] += self.dag.nodes[task_id]['ram_need']

            for task_id, (running_node, _) in self.task_status['running'].items():
                if running_node == node_id:
                    cpu_demands[node_id] += self.dag.nodes[task_id]['processing_need']
                    ram_demands[node_id] += self.dag.nodes[task_id]['ram_need']

            for task_id, (backup_node, _) in self.task_status['backup'].items():
                if backup_node == node_id:
                    cpu_demands[node_id] += self.dag.nodes[task_id]['processing_need']
                    ram_demands[node_id] += self.dag.nodes[task_id]['ram_need']

        cpu_normalized = np.zeros(self.num_nodes, dtype=np.float32)
        ram_normalized = np.zeros(self.num_nodes, dtype=np.float32)
        for node_id in range(self.num_nodes):
            if self.node_hierarchy['computing_power'][node_id] > 0:
                cpu_normalized[node_id] = cpu_demands[node_id] / self.node_hierarchy['computing_power'][node_id]
            if self.node_hierarchy['ram_capacity'][node_id] > 0:
                ram_normalized[node_id] = ram_demands[node_id] / self.node_hierarchy['ram_capacity'][node_id]

        avg_cpu = np.mean(cpu_normalized) if len(cpu_normalized) > 0 else 0
        avg_ram = np.mean(ram_normalized) if len(ram_normalized) > 0 else 0

        cpu_dev = np.sqrt(np.mean((cpu_normalized - avg_cpu) ** 2)) if len(cpu_normalized) > 0 else 0
        ram_dev = np.sqrt(np.mean((ram_normalized - avg_ram) ** 2)) if len(ram_normalized) > 0 else 0

        LD = cpu_dev + ram_dev

        return LD

    def _calculate_load_balance_reward(self, ld_before, ld_after):
        
        c = 2

        if ld_after <= ld_before:
            reward = ld_before - ld_after
        else:
            penalty = abs(ld_after - ld_before)
            reward = -c * penalty

        return reward

    def _calculate_average_reliability(self):
        
        if not self.task_allocations:
            return 0.0

        total_reliability = 0.0
        for task_id in self.task_allocations:
            node_id = self.task_allocations[task_id]
            total_reliability += self.node_reliabilities[node_id]

        return total_reliability / len(self.task_allocations)

    def _calculate_reliability_reward(self, avg_reliability_before, avg_reliability_after):
        
        c = 2

        if avg_reliability_after >= avg_reliability_before:
            reward = avg_reliability_after - avg_reliability_before
        else:
            penalty = abs(avg_reliability_after - avg_reliability_before)
            reward = -c * penalty

        return reward

    def _calculate_cloud_overuse_penalty(self, node_id):
        
        if node_id == 0:  
            tasks_on_cloud = 0
            for task_id in self.task_allocations:
                if self.task_allocations[task_id] == 0:
                    tasks_on_cloud += 1

            if tasks_on_cloud / max(1, len(self.task_allocations)) > 0.3:
                overuse_ratio = (tasks_on_cloud / max(1, len(self.task_allocations))) - 0.3
                return -2.0 * overuse_ratio 
        return 0.0

    def _calculate_diversity_reward(self, node_id):
        node_type = self.node_hierarchy['node_type'][node_id]

        used_types = set()
        for task_id in self.task_allocations:
            used_node = self.task_allocations[task_id]
            used_types.add(self.node_hierarchy['node_type'][used_node])

        if node_type not in used_types:
            return 0.3  
        return 0.0

    def _calculate_capacity_reward(self, node_id, task_ram_need, task_storage_need):
        if node_id >= len(self.node_ram_capacities) or node_id >= len(self.node_storage_capacities):
            return 0.0

        ram_ratio = self.remaining_ram[node_id] / max(1e-6, self.node_ram_capacities[node_id])
        storage_ratio = self.remaining_storage[node_id] / max(1e-6, self.node_storage_capacities[node_id])

        if ram_ratio < 0.1 or storage_ratio < 0.1:  
            return -0.5
        elif 0.3 <= ram_ratio <= 0.7 and 0.3 <= storage_ratio <= 0.7:
            return 0.2
        return 0.0

    def _calculate_deadline_reward(self):
        if self.makespan <= self.current_deadline:
            if self.current_priority == 1:
                return 3.0 * (1.0 - self.makespan / self.current_deadline)
            elif self.current_priority == 2:
                return 2.0 * (1.0 - self.makespan / self.current_deadline)
            else:
                return 1.0 * (1.0 - self.makespan / self.current_deadline)
        else:
            time_excess = self.makespan - self.current_deadline
            return -5.0 * (time_excess / self.current_deadline)

    def _calculate_reliability_objective_f3(self):
        
        if not self.task_allocations:
            return 0.0

        total_inverse_reliability = 0.0
        for task_id in self.task_allocations:
            node_id = self.task_allocations[task_id]
            reliability = max(1e-6, self.node_reliabilities[node_id])
            total_inverse_reliability += 1.0 / reliability

        f3 = total_inverse_reliability / len(self.task_allocations)

        return f3

    def reset(self):
       
        self.deadline_violations_by_priority = {1: 0, 2: 0, 3: 0}
        self.episode_timing_history = []  

        self.current_dag_idx = 0
        self.dag = copy.deepcopy(self.dags[self.current_dag_idx])
        self.current_priority = self.dag_priorities[self.current_dag_idx]
        self.current_deadline = self.dag_deadlines[self.current_dag_idx]
        self.current_arrival_time = self.dag_arrival_times[self.current_dag_idx]
        self.total_reward = 0
        self.dag_rewards = []
        self.last_completed_dag_info = None  
        self.episode_allocation_history = []  
        self.total_response_time = 0.0  
        self.total_load_deviation = 0.0  
        self.total_reliability_score = 0.0  
        self.load_deviation_history = [] 
        self.reliability_history = [] 

        self.backup_allocations = {}  

        self.node_ram_capacities = self.node_hierarchy['ram_capacity'].copy()
        self.node_storage_capacities = self.node_hierarchy['storage_capacity'].copy()
        self.node_computing_powers = self.node_hierarchy['computing_power'].copy()

        self.remaining_ram = self.node_ram_capacities.copy()
        self.remaining_storage = self.node_storage_capacities.copy()
        self.node_reliabilities = self.node_hierarchy['reliability'].copy()

        self.node_queues = {}
        for node_id in range(self.num_nodes):
            self.node_queues[node_id] = {
                1: deque(),  
                2: deque(),  
                3: deque()   
            }

        self.task_status = {
            'waiting': set(range(self.num_tasks)),
            'ready': set(),
            'running': {},
            'backup': {},  
            'completed': set(),
            'completed_backup': set(), 
            'primary_completion_times': {}  
        }

        self.task_allocations = {}

       
        self.current_time = self.current_arrival_time
        self.makespan = 0
        self.dag_start_time = self.current_arrival_time  
        self.dag_absolute_completion_time = None  

        self._update_task_states()

        return self._get_obs()

    def _load_next_dag(self):
        self.current_dag_idx += 1
        if self.current_dag_idx < self.num_dags:
            self.dag = copy.deepcopy(self.dags[self.current_dag_idx])
            self.current_priority = self.dag_priorities[self.current_dag_idx]
            self.current_deadline = self.dag_deadlines[self.current_dag_idx]
            self.current_arrival_time = self.dag_arrival_times[self.current_dag_idx]

           
            self.current_time = max(self.current_time, self.current_arrival_time)

            self.remaining_ram = self.node_ram_capacities.copy()
            self.remaining_storage = self.node_storage_capacities.copy()

            for node_id in range(self.num_nodes):
                self.node_queues[node_id] = {
                    1: deque(),
                    2: deque(),
                    3: deque()
                }

            self.task_status = {
                'waiting': set(range(self.num_tasks)),
                'ready': set(),
                'running': {},
                'backup': {},
                'completed': set(),
                'completed_backup': set(),
                'primary_completion_times': {}
            }

            self.task_allocations = {}
            self.backup_allocations = {} 
            self.makespan = 0
            self.dag_start_time = self.current_time
            self.total_response_time = 0.0  
            self.total_load_deviation = 0.0  
            self.total_reliability_score = 0.0  
            self.load_deviation_history = []  
            self.reliability_history = []  

            self._update_task_states()

            return True
        return False

    def _update_task_states(self):
       
        for task_id in list(self.task_status['waiting']):
            predecessors = list(self.dag.predecessors(task_id))
            if all(pred in self.task_status['completed'] for pred in predecessors):
                self.task_status['waiting'].remove(task_id)
                self.task_status['ready'].add(task_id)

    def _get_obs(self):
       
        ram_status = self.remaining_ram.copy()

        storage_status = self.remaining_storage.copy()

        reliability_status = self.node_reliabilities.copy()

        queue_status = np.zeros(self.num_nodes * 3, dtype=np.float32)  
        for node_id in range(self.num_nodes):
            for priority in [1, 2, 3]:
                queue_status[node_id * 3 + (priority - 1)] = len(self.node_queues[node_id][priority])

        ready_tasks = np.zeros(self.num_tasks, dtype=np.float32)
        for task_id in self.task_status['ready']:
            ready_tasks[task_id] = 1.0

        task_info = np.zeros(3, dtype=np.float32)
        if self.task_status['ready']:
            sample_task = next(iter(self.task_status['ready']))
            task_info[0] = self.dag.nodes[sample_task]['ram_need']
            task_info[1] = self.dag.nodes[sample_task]['storage_need']
            task_info[2] = self.dag.nodes[sample_task]['deadline']

        
        progress_info = np.array([
            self.current_dag_idx / self.num_dags,
            len(self.task_status['completed']) / self.num_tasks,
            self.current_priority / 3.0,  
            self.current_time / self.current_deadline if self.current_deadline > 0 else 0, 
            len(self.backup_allocations) / self.num_tasks 
        ], dtype=np.float32)

        network_info = np.array([
            np.mean(self.delay_matrix),
            np.mean(self.bandwidth_matrix)
        ], dtype=np.float32)

        obs = np.concatenate([
            ram_status, storage_status, reliability_status,
            queue_status, ready_tasks, task_info,
            progress_info, network_info
        ]).astype(np.float32)

        return obs

    def step(self, action):
        reward = 0.0
        done = False
        dag_completed = False

        ld_before = self._calculate_load_deviation()

        avg_reliability_before = self._calculate_average_reliability()

        if not self.task_status['ready']:
            reward = -0.1
            self.current_time += 0.1  
            ld_after = ld_before  
            avg_reliability_after = avg_reliability_before  
        else:
            task_id = next(iter(self.task_status['ready']))

            if action == self.num_nodes: 
                reward = -0.5
                ld_after = ld_before 
                avg_reliability_after = avg_reliability_before  
            else:
                ram_need = self.dag.nodes[task_id]['ram_need']
                storage_need = self.dag.nodes[task_id]['storage_need']

                if (self.remaining_ram[action] >= ram_need and
                    self.remaining_storage[action] >= storage_need):

                    response_time = self._calculate_response_time_paper(task_id, action)

                    response_reward = -response_time * 10.0

                    self.total_response_time += response_time

                    self.remaining_ram[action] -= ram_need
                    self.remaining_storage[action] -= storage_need
                    self.task_status['ready'].remove(task_id)

                    self.node_queues[action][self.current_priority].append(task_id)

                    execution_time = self._calculate_execution_time(task_id, action, self.current_priority)

                    actual_start_time = self.current_time + execution_time
                    self.task_status['running'][task_id] = (action, actual_start_time)
                    self.task_allocations[task_id] = action

                    
                    if self.current_priority == 1:  
                        self._create_backup_for_priority_1(task_id, action)
                    elif self.current_priority == 2:  
                        self._create_backup_for_priority_2(task_id, action)

                    ld_after = self._calculate_load_deviation()

                    avg_reliability_after = self._calculate_average_reliability()

                    load_balance_reward = self._calculate_load_balance_reward(ld_before, ld_after)

                    reliability_reward = self._calculate_reliability_reward(avg_reliability_before, avg_reliability_after)

                    cloud_penalty = self._calculate_cloud_overuse_penalty(action)
                    diversity_reward = self._calculate_diversity_reward(action)
                    capacity_reward = self._calculate_capacity_reward(action, ram_need, storage_need)

                    weights = PRIORITY_WEIGHTS[self.current_priority]
                    w1, w2, w3 = weights['response_time'], weights['load_balance'], weights['reliability']

                    scaled_load_balance_reward = load_balance_reward * 8.0

                    scaled_reliability_reward = reliability_reward * 5.0

                    combined_reward = (
                        w1 * response_reward +
                        w2 * scaled_load_balance_reward +
                        w3 * scaled_reliability_reward +
                        cloud_penalty +
                        capacity_reward
                    )

                    reward = combined_reward
                    self.total_load_deviation += ld_after
                    self.load_deviation_history.append(ld_after)

                    current_f3 = self._calculate_reliability_objective_f3()
                    self.total_reliability_score += current_f3
                    self.reliability_history.append(current_f3)

                else:
                    reward = -1.0
                    ld_after = ld_before 
                    avg_reliability_after = avg_reliability_before  

        self._update_running_tasks()

        self._update_task_states()

        if len(self.task_status['completed']) == self.num_tasks:
            dag_completed = True

            self.dag_absolute_completion_time = self.current_time

            dag_execution_time = self.dag_absolute_completion_time - self.dag_start_time

            timing_info = {
                'dag_idx': self.current_dag_idx,
                'priority': self.current_priority,
                'arrival_time': self.current_arrival_time,
                'start_time': self.dag_start_time,
                'completion_time': self.dag_absolute_completion_time,
                'execution_time': dag_execution_time,
                'deadline': self.current_deadline,
                'deadline_met': dag_execution_time <= self.current_deadline,
                'makespan': self.makespan
            }
            self.episode_timing_history.append(timing_info)

            if dag_execution_time > self.current_deadline:
                self.deadline_violations_by_priority[self.current_priority] += 1

            avg_load_deviation = np.mean(self.load_deviation_history) if self.load_deviation_history else 0

            avg_reliability_f3 = np.mean(self.reliability_history) if self.reliability_history else 0

            weights = PRIORITY_WEIGHTS[self.current_priority]

           
            f1_reward = -self.total_response_time * 2.0  
            f2_reward = -avg_load_deviation * 10.0  
            f3_reward = -avg_reliability_f3 * 5.0  
            deadline_reward = self._calculate_deadline_reward() 

            w1_final = weights['response_time'] * 0.7  
            w2_final = weights['load_balance'] * 0.7  
            w3_final = weights['reliability'] * 0.7    


            final_reward = (
                w1_final * f1_reward +
                w2_final * f2_reward +
                w3_final * f3_reward
            )

           
            reward += final_reward
            self.dag_rewards.append(final_reward)

            avg_node_reliability = np.mean(self.node_reliabilities)

            primary_reliabilities = []
            backup_reliabilities = []
            primary_backup_pairs = []

            task_allocations_details = []
            for task_id in range(self.num_tasks):
                primary_node = self.task_allocations.get(task_id, None)
                primary_reliability = self.node_reliabilities[primary_node] if primary_node is not None else None

                backup_node = self.backup_allocations.get(task_id, None)
                backup_reliability = self.node_reliabilities[backup_node] if backup_node is not None else None

                has_backup = backup_node is not None

                task_allocations_details.append({
                    'task_id': task_id,
                    'primary_node': primary_node,
                    'primary_reliability': primary_reliability,
                    'backup_node': backup_node,
                    'backup_reliability': backup_reliability,
                    'has_backup': has_backup
                })

                if has_backup and primary_node is not None:
                    primary_reliabilities.append(primary_reliability)
                    backup_reliabilities.append(backup_reliability)
                    primary_backup_pairs.append({
                        'task_id': task_id,
                        'primary_node': primary_node,
                        'primary_reliability': primary_reliability,
                        'backup_node': backup_node,
                        'backup_reliability': backup_reliability
                    })

            self.last_completed_dag_info = {
                'dag_idx': self.current_dag_idx,
                'priority': self.current_priority,
                'weights': weights, 
                'num_tasks_with_backup': len(self.backup_allocations),
                'avg_node_reliability': avg_node_reliability,
                'primary_reliabilities': primary_reliabilities,
                'backup_reliabilities': backup_reliabilities,
                'primary_backup_pairs': primary_backup_pairs,
                'task_allocations_details': task_allocations_details,
                'makespan': self.makespan,
                'deadline': self.current_deadline,
                'deadline_met': dag_execution_time <= self.current_deadline,
                'deadline_reward': deadline_reward,
                'total_response_time': self.total_response_time,
                'avg_response_time': self.total_response_time / self.num_tasks if self.num_tasks > 0 else 0,
                'avg_load_deviation': avg_load_deviation,
                'total_load_deviation': self.total_load_deviation,
                'avg_reliability_f3': avg_reliability_f3,
                'total_reliability_score': self.total_reliability_score,
                'avg_reliability_before': avg_reliability_before,
                'avg_reliability_after': avg_reliability_after,
                'arrival_time': self.current_arrival_time,
                'start_time': self.dag_start_time,
                'completion_time': self.dag_absolute_completion_time,
                'execution_time': dag_execution_time
            }

            self.episode_allocation_history.append(self.last_completed_dag_info)

            if not self._load_next_dag():
                done = True

        self.total_reward += reward

        return self._get_obs(), reward, done, dag_completed

    def _update_running_tasks(self):
        completed_tasks = []
        completed_backup_tasks = []

        for task_id, (node_id, completion_time) in self.task_status['running'].items():
            if self.current_time >= completion_time:
                completed_tasks.append(task_id)

                ram_need = self.dag.nodes[task_id]['ram_need']
                storage_need = self.dag.nodes[task_id]['storage_need']
                self.remaining_ram[node_id] += ram_need
                self.remaining_storage[node_id] += storage_need

                if task_id in self.node_queues[node_id][self.current_priority]:
                    self.node_queues[node_id][self.current_priority].remove(task_id)

                self.makespan = max(self.makespan, self.current_time - self.dag_start_time)
                self.task_status['primary_completion_times'][task_id] = self.current_time

        for task_id, (node_id, completion_time) in self.task_status['backup'].items():
            if self.current_time >= completion_time:
                completed_backup_tasks.append(task_id)

                ram_need = self.dag.nodes[task_id]['ram_need']
                storage_need = self.dag.nodes[task_id]['storage_need']
                self.remaining_ram[node_id] += ram_need
                self.remaining_storage[node_id] += storage_need

                if task_id in self.node_queues[node_id][self.current_priority]:
                    self.node_queues[node_id][self.current_priority].remove(task_id)

        for task_id in completed_tasks:
            del self.task_status['running'][task_id]
            self.task_status['completed'].add(task_id)

        for task_id in completed_backup_tasks:
            del self.task_status['backup'][task_id]
            self.task_status['completed_backup'].add(task_id)

        self.current_time += 0.1

    def get_episode_timing_report(self):
        report = f"\n{'='*100}"
        report += f"\nEPISODE TIMING REPORT - DAG ARRIVAL AND COMPLETION TIMES"
        report += f"\n{'='*100}"

        for timing_info in self.episode_timing_history:
            priority_names = {1: "High", 2: "Medium", 3: "Low"}
            status = "✓ MET" if timing_info['deadline_met'] else " VIOLATED"

            report += f"\nDAG {timing_info['dag_idx'] + 1} (Priority: {priority_names[timing_info['priority']]}):"
            report += f"\n  Arrival Time: {timing_info['arrival_time']:.2f}"
            report += f"\n  Start Time: {timing_info['start_time']:.2f}"
            report += f"\n  Completion Time: {timing_info['completion_time']:.2f}"
            report += f"\n  Execution Time: {timing_info['execution_time']:.2f}"
            report += f"\n  Deadline: {timing_info['deadline']:.2f}"
            report += f"\n  Status: {status}"
            report += f"\n  Makespan (relative): {timing_info['makespan']:.2f}"
            report += f"\n"

        report += f"\n{'='*80}"
        report += f"\nDEADLINE VIOLATION STATISTICS BY PRIORITY"
        report += f"\n{'='*80}"

        total_dags = len(self.episode_timing_history)
        for priority in [1, 2, 3]:
            priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
            violations = self.deadline_violations_by_priority[priority]
            dags_in_priority = sum(1 for d in self.episode_timing_history if d['priority'] == priority)

            if dags_in_priority > 0:
                violation_rate = (violations / dags_in_priority) * 100
                report += f"\nPriority {priority_name}: {violations}/{dags_in_priority} violated ({violation_rate:.1f}%)"

        total_violations = sum(self.deadline_violations_by_priority.values())
        total_violation_rate = (total_violations / total_dags) * 100 if total_dags > 0 else 0

        report += f"\n\nOverall: {total_violations}/{total_dags} DAGs violated deadlines ({total_violation_rate:.1f}%)"

        report += f"\n\n{'='*80}"
        report += f"\nAVERAGE EXECUTION TIME BY PRIORITY"
        report += f"\n{'='*80}"

        for priority in [1, 2, 3]:
            priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
            priority_times = [d['execution_time'] for d in self.episode_timing_history if d['priority'] == priority]

            if priority_times:
                avg_time = np.mean(priority_times)
                std_time = np.std(priority_times)
                report += f"\nPriority {priority_name}: {avg_time:.2f} ± {std_time:.2f}"

        return report

    def render(self):
        priority_names = {1: "High", 2: "Medium", 3: "Low"}

        print(f"\n{'='*80}")
        print(f"Diabetes Management System - Hierarchical Infrastructure")
        print(f"{'='*80}")
        print(f"DAG {self.current_dag_idx + 1}/{self.num_dags} (Priority: {priority_names[self.current_priority]})")
        print(f"Arrival Time: {self.current_arrival_time:.2f}, Current Time: {self.current_time:.2f}")
        print(f"Deadline: {self.current_deadline:.2f}, Time Since Arrival: {self.current_time - self.current_arrival_time:.2f}")

        weights = PRIORITY_WEIGHTS[self.current_priority]
        print(f"Weights - Response: {weights['response_time']:.2f}, Load Balance: {weights['load_balance']:.2f}, Reliability: {weights['reliability']:.2f}")

        print(f"Time: {self.current_time:.2f}, Makespan: {self.makespan:.2f}")
        print(f"Total Response Time (F1): {self.total_response_time:.2f}")
        print(f"Average Response Time per Task: {self.total_response_time/self.num_tasks:.2f}" if self.num_tasks > 0 else "No tasks")
        print(f"Total Load Deviation (F2): {self.total_load_deviation:.4f}")
        print(f"Current Load Deviation: {self._calculate_load_deviation():.4f}")
        print(f"Total Reliability Score (F3): {self.total_reliability_score:.4f}")
        print(f"Current Average Reliability: {self._calculate_average_reliability():.4f}")

        node_type_distribution = {}
        for task_id in self.task_allocations:
            node_id = self.task_allocations[task_id]
            node_type = self.node_hierarchy['node_type'][node_id]
            node_type_distribution[node_type] = node_type_distribution.get(node_type, 0) + 1

        print(f"Task Allocation Distribution:")
        for node_type in ['cloud', 'fog2', 'fog1', 'edge']:
            count = node_type_distribution.get(node_type, 0)
            percentage = (count / max(1, len(self.task_allocations))) * 100
            print(f"  {node_type}: {count} tasks ({percentage:.1f}%)")

        print(f"Completed tasks: {len(self.task_status['completed'])}/{self.num_tasks}")
        print(f"Backup tasks: {len(self.backup_allocations)} completed: {len(self.task_status['completed_backup'])}")
        print(f"Ready tasks: {len(self.task_status['ready'])}")
        print(f"Running tasks: {len(self.task_status['running'])}")

        time_since_arrival = self.current_time - self.current_arrival_time
        if time_since_arrival <= self.current_deadline:
            print(f"Deadline status: On time ({time_since_arrival:.2f}/{self.current_deadline:.2f}, {time_since_arrival/self.current_deadline*100:.1f}% elapsed)")
        else:
            print(f"Deadline status: Violated ({time_since_arrival - self.current_deadline:.2f} time units overdue)")

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.9, clip_epsilon=0.1, entropy_coef=0.01):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.003)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=0.01)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

    def select_action(self, state, episode):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.policy_net(state)

        
        dist = Categorical(probs)
        action = dist.sample()

        entropy = dist.entropy()

        return action.item(), dist.log_prob(action), entropy

    def update(self, states, actions, log_probs_old, rewards, dones, entropies):
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns).to(device)
        values = self.value_net(states).squeeze()
        advantages = returns - values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        value_loss = nn.MSELoss()(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        probs = self.policy_net(states)
        dist = Categorical(probs)
        log_probs_new = dist.log_prob(actions)

        ratio = torch.exp(log_probs_new - log_probs_old)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        entropy = dist.entropy().mean()

        policy_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_coef * entropy

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

def train_ppo(env, agent, num_episodes=500):
    episode_rewards = []
    dag_completion_times = []
    deadline_violations = [] 
    backup_rates = []  
    response_times = []  
    load_deviations = []  
    reliability_scores = []  
    cloud_usage_rates = []  
    priority_rewards = {1: [], 2: [], 3: []}  
    priority_weights_history = {1: [], 2: [], 3: []}  
    recent_rewards = deque(maxlen=100)

    
    episode_arrival_times = []  
    episode_start_times = []  
    episode_completion_times = []  
    episode_execution_times = []  

    backup_stats = {1: [], 2: [], 3: []}

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        dags_completed = 0
        violations = 0
        total_response_time = 0.0
        total_load_deviation = 0.0
        total_reliability_score = 0.0
        total_cloud_usage = 0.0

        episode_dags_info = []

        states, actions, log_probs, rewards, dones, entropies = [], [], [], [], [], []

        while not done:
            action, log_prob, entropy = agent.select_action(state, episode)
            next_state, reward, done, dag_completed = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            entropies.append(entropy.item())

            state = next_state
            total_reward += reward

            if dag_completed:
                dags_completed += 1

                if env.last_completed_dag_info is not None:
                    dag_info = env.last_completed_dag_info
                    episode_dags_info.append(dag_info)

                    backup_stats[dag_info['priority']].append(dag_info['num_tasks_with_backup'])

                    priority_weights_history[dag_info['priority']].append(dag_info['weights'])

                    total_response_time += dag_info['total_response_time']
                    total_load_deviation += dag_info['total_load_deviation']
                    total_reliability_score += dag_info['total_reliability_score']

                    cloud_tasks = 0
                    for task_detail in dag_info['task_allocations_details']:
                        if task_detail['primary_node'] == 0: 
                            cloud_tasks += 1
                    total_cloud_usage += cloud_tasks / env.num_tasks

                    env.last_completed_dag_info = None

                if dag_info['execution_time'] > dag_info['deadline']:
                    violations += 1
                backup_rate = len(env.backup_allocations) / env.num_tasks if env.num_tasks > 0 else 0
                backup_rates.append(backup_rate)
                priority_rewards[env.current_priority].append(reward)

        if states:
            agent.update(states, actions, log_probs, rewards, dones, entropies)

        episode_rewards.append(total_reward)
        dag_completion_times.append(dags_completed)
        deadline_violations.append(violations)
        response_times.append(total_response_time / dags_completed if dags_completed > 0 else 0)
        load_deviations.append(total_load_deviation / dags_completed if dags_completed > 0 else 0)
        reliability_scores.append(total_reliability_score / dags_completed if dags_completed > 0 else 0)
        cloud_usage_rates.append(total_cloud_usage / dags_completed if dags_completed > 0 else 0)
        recent_rewards.append(total_reward)

        episode_timing_summary = {
            'episode': episode,
            'arrival_times': env.original_arrival_times,
            'timing_history': env.episode_timing_history,
            'deadline_violations': env.deadline_violations_by_priority
        }

        if episode % 10 == 0 or episode == num_episodes - 1:
            print(env.get_episode_timing_report())

        if episode == num_episodes - 1:
            print(f"\n{'='*100}")
            print(f"FINAL EPISODE - COMPLETE TASK ALLOCATION REPORT")
            print(f"{'='*100}")

            priority_names = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}

            dags_by_priority = {1: [], 2: [], 3: []}
            for dag_info in env.episode_allocation_history:
                dags_by_priority[dag_info['priority']].append(dag_info)

            for priority in [1, 2, 3]:
                if dags_by_priority[priority]:
                    print(f"\n{'='*80}")
                    print(f"PRIORITY {priority_names[priority]} DAGs - DETAILED ALLOCATION")
                    print(f"{'='*80}")

                    for dag_idx, dag_info in enumerate(dags_by_priority[priority]):
                        print(f"\nDAG #{dag_idx + 1} (Global Index: {dag_info['dag_idx'] + 1})")
                        print(f"Arrival Time: {dag_info['arrival_time']:.2f}, Start Time: {dag_info['start_time']:.2f}")
                        print(f"Completion Time: {dag_info['completion_time']:.2f}, Execution Time: {dag_info['execution_time']:.2f}")
                        print(f"Weights Used: Response={dag_info['weights']['response_time']:.2f}, "
                              f"Load Balance={dag_info['weights']['load_balance']:.2f}, "
                              f"Reliability={dag_info['weights']['reliability']:.2f}")
                        print(f"Makespan: {dag_info['makespan']:.2f}, Deadline: {dag_info['deadline']:.2f}, "
                              f"Status: {'✓ MET' if dag_info['deadline_met'] else '✗ VIOLATED'}")
                        print(f"Total Response Time (F1): {dag_info['total_response_time']:.2f}")
                        print(f"Average Response Time: {dag_info['avg_response_time']:.2f}")
                        print(f"Average Load Deviation (F2): {dag_info['avg_load_deviation']:.4f}")
                        print(f"Average Reliability Score (F3): {dag_info['avg_reliability_f3']:.4f}")
                        print(f"Deadline Reward: {dag_info['deadline_reward']:.2f}")
                        print(f"Reliability Before/After: {dag_info['avg_reliability_before']:.4f} → {dag_info['avg_reliability_after']:.4f}")

                        node_type_distribution = {}
                        for task_detail in dag_info['task_allocations_details']:
                            primary_node = task_detail['primary_node']
                            if primary_node is not None:
                                node_type = env.node_hierarchy['node_type'][primary_node]
                                node_type_distribution[node_type] = node_type_distribution.get(node_type, 0) + 1

                        print(f"Task Allocation Distribution:")
                        for node_type in ['cloud', 'fog2', 'fog1', 'edge']:
                            count = node_type_distribution.get(node_type, 0)
                            percentage = (count / env.num_tasks) * 100
                            print(f"  {node_type}: {count} tasks ({percentage:.1f}%)")

                        print(f"Tasks with SMART Backup: {dag_info['num_tasks_with_backup']}/{env.num_tasks} "
                              f"({dag_info['num_tasks_with_backup']/env.num_tasks*100:.1f}%)")
                        print(f"Average Node Reliability: {dag_info['avg_node_reliability']:.4f}")

                        if dag_info['num_tasks_with_backup'] > 0:
                            avg_primary = np.mean(dag_info['primary_reliabilities']) if dag_info['primary_reliabilities'] else 0
                            avg_backup = np.mean(dag_info['backup_reliabilities']) if dag_info['backup_reliabilities'] else 0
                            print(f"Avg Primary Reliability: {avg_primary:.4f}, Avg Backup Reliability: {avg_backup:.4f}")

        print(f"\n{'='*80}")
        print(f"Episode {episode + 1} - Summary:")
        print(f"{'='*80}")

        priority_counts = {1: 0, 2: 0, 3: 0}
        total_backup_tasks = 0
        total_response_time_episode = 0
        total_load_deviation_episode = 0
        total_reliability_score_episode = 0
        total_cloud_usage_episode = 0

        for i, dag_info in enumerate(episode_dags_info):
            priority_counts[dag_info['priority']] += 1
            total_backup_tasks += dag_info['num_tasks_with_backup']
            total_response_time_episode += dag_info['total_response_time']
            total_load_deviation_episode += dag_info['total_load_deviation']
            total_reliability_score_episode += dag_info['total_reliability_score']

            cloud_tasks = 0
            for task_detail in dag_info['task_allocations_details']:
                if task_detail['primary_node'] == 0:  
                    cloud_tasks += 1
            total_cloud_usage_episode += cloud_tasks / env.num_tasks

        print(f"Total Reward: {total_reward:.2f}")
        print(f"DAGs Completed: {dags_completed}/{env.num_dags}")
        print(f"Priority Distribution: High={priority_counts[1]}, Medium={priority_counts[2]}, Low={priority_counts[3]}")
        print(f"Total SMART Backup Tasks Created: {total_backup_tasks}")
        print(f"Total Response Time (F1): {total_response_time_episode:.2f}")
        print(f"Average Response Time per DAG: {total_response_time_episode/dags_completed:.2f}" if dags_completed > 0 else "No DAGs")
        print(f"Total Load Deviation (F2): {total_load_deviation_episode:.4f}")
        print(f"Average Load Deviation per DAG: {total_load_deviation_episode/dags_completed:.4f}" if dags_completed > 0 else "No DAGs")
        print(f"Total Reliability Score (F3): {total_reliability_score_episode:.4f}")
        print(f"Average Reliability Score per DAG: {total_reliability_score_episode/dags_completed:.4f}" if dags_completed > 0 else "No DAGs")
        print(f"Average Cloud Usage per DAG: {total_cloud_usage_episode/dags_completed*100:.1f}%" if dags_completed > 0 else "No DAGs")

        print(f"\nSMART Backup Statistics by Priority:")
        for priority in [1, 2, 3]:
            if backup_stats[priority]:
                avg_backup = np.mean(backup_stats[priority])
                print(f"  Priority {priority}: {avg_backup:.1f} backup tasks on average")

        print(f"Deadline Violations: {violations}/{dags_completed} ({violations/dags_completed*100:.1f}%)")

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_dags = np.mean(dag_completion_times[-50:]) if len(dag_completion_times) >= 50 else dags_completed
            avg_violations = np.mean(deadline_violations[-50:]) if len(deadline_violations) >= 50 else violations
            avg_backup = np.mean(backup_rates[-50:]) if len(backup_rates) >= 50 else 0
            avg_response = np.mean(response_times[-50:]) if len(response_times) >= 50 else 0
            avg_load_dev = np.mean(load_deviations[-50:]) if len(load_deviations) >= 50 else 0
            avg_reliability = np.mean(reliability_scores[-50:]) if len(reliability_scores) >= 50 else 0
            avg_cloud_usage = np.mean(cloud_usage_rates[-50:]) if len(cloud_usage_rates) >= 50 else 0

            avg_priority_rewards = {}
            for priority in [1, 2, 3]:
                if priority_rewards[priority]:
                    avg_priority_rewards[priority] = np.mean(priority_rewards[priority])
                else:
                    avg_priority_rewards[priority] = 0

            print(f"\nTraining Progress (Episode {episode + 1}):")
            print(f"  Recent Average Reward: {avg_reward:.2f}")
            print(f"  Recent Average DAGs Completed: {avg_dags:.2f}")
            print(f"  Recent Average Response Time (F1): {avg_response:.2f}")
            print(f"  Recent Average Load Deviation (F2): {avg_load_dev:.4f}")
            print(f"  Recent Average Reliability Score (F3): {avg_reliability:.4f}")
            print(f"  Recent Average Cloud Usage: {avg_cloud_usage*100:.1f}%")
            print(f"  Recent Average Deadline Violations: {avg_violations:.2f}")
            print(f"  Recent Average SMART Backup Rate: {avg_backup:.2%}")
            print(f"  Rewards by Priority - High: {avg_priority_rewards[1]:.2f}, Medium: {avg_priority_rewards[2]:.2f}, Low: {avg_priority_rewards[3]:.2f}")

            print(f"  Priority Weights Used:")
            for priority in [1, 2, 3]:
                if priority_weights_history[priority]:
                    weights_list = priority_weights_history[priority]
                    avg_response_w = np.mean([w['response_time'] for w in weights_list])
                    avg_load_w = np.mean([w['load_balance'] for w in weights_list])
                    avg_reliability_w = np.mean([w['reliability'] for w in weights_list])
                    print(f"    Priority {priority}: Response={avg_response_w:.2f}, Load={avg_load_w:.2f}, Reliability={avg_reliability_w:.2f}")

    return episode_rewards, dag_completion_times, deadline_violations, backup_rates, response_times, load_deviations, reliability_scores, cloud_usage_rates, priority_rewards

def train_ppo_quick(env, agent, num_episodes=500):
    episode_rewards = []
    dag_completion_times = []
    deadline_violations = []
    backup_rates = []
    response_times = []
    load_deviations = []
    reliability_scores = []
    cloud_usage_rates = []
    priority_rewards = {1: [], 2: [], 3: []}

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        dags_completed = 0
        violations = 0
        total_response_time = 0.0
        total_load_deviation = 0.0
        total_reliability_score = 0.0
        total_cloud_usage = 0.0

        episode_dags_info = []
        states, actions, log_probs, rewards, dones, entropies = [], [], [], [], [], []

        while not done:
            action, log_prob, entropy = agent.select_action(state, episode)
            next_state, reward, done, dag_completed = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            entropies.append(entropy.item())

            state = next_state
            total_reward += reward

            if dag_completed:
                dags_completed += 1

                if env.last_completed_dag_info is not None:
                    dag_info = env.last_completed_dag_info
                    episode_dags_info.append(dag_info)

                    total_response_time += dag_info['total_response_time']
                    total_load_deviation += dag_info['total_load_deviation']
                    total_reliability_score += dag_info['total_reliability_score']

                    cloud_tasks = 0
                    for task_detail in dag_info['task_allocations_details']:
                        if task_detail['primary_node'] == 0:
                            cloud_tasks += 1
                    total_cloud_usage += cloud_tasks / env.num_tasks

                    env.last_completed_dag_info = None

                if dag_info['execution_time'] > dag_info['deadline']:
                    violations += 1

                backup_rate = len(env.backup_allocations) / env.num_tasks if env.num_tasks > 0 else 0
                backup_rates.append(backup_rate)
                priority_rewards[env.current_priority].append(reward)

        if states:
            agent.update(states, actions, log_probs, rewards, dones, entropies)

        episode_rewards.append(total_reward)
        dag_completion_times.append(dags_completed)
        deadline_violations.append(violations)
        response_times.append(total_response_time / dags_completed if dags_completed > 0 else 0)
        load_deviations.append(total_load_deviation / dags_completed if dags_completed > 0 else 0)
        reliability_scores.append(total_reliability_score / dags_completed if dags_completed > 0 else 0)
        cloud_usage_rates.append(total_cloud_usage / dags_completed if dags_completed > 0 else 0)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {np.mean(episode_rewards[-50:]):.2f}, "
                  f"DAGs Completed: {dags_completed}")

    return episode_rewards, dag_completion_times, deadline_violations, backup_rates, response_times, load_deviations, reliability_scores, cloud_usage_rates, priority_rewards

if __name__ == "__main__":

    num_dags = 50
    num_episodes = 100

    clip_values = [0.1, 0.2, 0.3, 0.4]

    all_results = {}

    print(f"\n{'='*80}")
    print(f"COMPARING DIFFERENT CLIP_EPSILON VALUES (γ=0.9)")
    print(f"{'='*80}")

    for clip_val in clip_values:
        print(f"\n{'='*80}")
        print(f"Training with clip_epsilon = {clip_val}, γ=0.9")
        print(f"{'='*80}")

        env = DiabetesDAGEnv(num_dags=num_dags)

        state_dim = env.num_nodes * 3 + env.num_nodes * 3 + env.num_tasks + 3 + 5 + 2
        action_dim = env.num_nodes + 1

        agent = PPO(state_dim, action_dim, lr=3e-4, gamma=0.9, clip_epsilon=clip_val, entropy_coef=0.1)

        print(f"Starting training with {num_episodes} episodes...")
        start_time = time.time()
        results = train_ppo_quick(env, agent, num_episodes=num_episodes)
        end_time = time.time()

        print(f"Training with clip_epsilon = {clip_val} completed in {end_time - start_time:.2f} seconds")

       
        all_results[clip_val] = results

   
    plt.figure(figsize=(15, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':']

    final_values = {}

    for i, (clip_val, results) in enumerate(all_results.items()):
        rewards, dags_completed, violations, backup_rates, response_times, load_deviations, reliability_scores, cloud_usage_rates, priority_rewards = results

        if len(rewards) >= 20:
            final_avg = np.mean(rewards[-20:])
        else:
            final_avg = np.mean(rewards)
        final_values[clip_val] = final_avg

        line_style = line_styles[i % len(line_styles)]

        plt.plot(range(len(rewards)), rewards,
                 label=f'ε={clip_val} (Final: {final_avg:.2f})',
                 color=colors[i % len(colors)],
                 linestyle=line_style,
                 linewidth=1.5,
                 alpha=0.8)


    plt.title('Reward Convergence for Different Clipping Values (γ=0.9)\nRaw Reward Values (No Moving Average)',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)

    plt.grid(True, alpha=0.3, linestyle='--')


    plt.xlim(left=0, right=num_episodes - 1)


    legend = plt.legend(fontsize=11, loc='best', framealpha=0.9, shadow=True)
    legend.get_frame().set_facecolor('#f5f5f5')

 
    plt.figtext(0.5, 0.01,
                f'Raw reward values for all {num_episodes} episodes | Parameters: {num_dags} DAGs per episode, γ=0.9',
                ha='center', fontsize=10, style='italic')


    best_clip = max(final_values.items(), key=lambda x: x[1])
    best_text = f'Best: ε={best_clip[0]} (Avg Reward: {best_clip[1]:.2f})'
    plt.figtext(0.15, 0.02, best_text, fontsize=10, fontweight='bold', color='green')

   
    reward_levels = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
    for level in reward_levels:
        plt.axhline(y=level, color='gray', linestyle=':', alpha=0.2, linewidth=0.5)

   
    plt.figtext(0.15, 0.94, f'Clip ε Comparison (γ=0.9)',
                fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

 
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  
    plt.savefig('clip_epsilon_raw_rewards_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()

    plt.figure(figsize=(15, 6))

   
    plt.subplot(1, 2, 1)
    for i, (clip_val, results) in enumerate(all_results.items()):
        rewards, _, _, _, _, _, _, _, _ = results
        line_style = line_styles[i % len(line_styles)]
        plt.plot(range(len(rewards)), rewards,
                 label=f'ε={clip_val}',
                 color=colors[i % len(colors)],
                 linestyle=line_style,
                 linewidth=1.2,
                 alpha=0.7)

    plt.title('Full Training (All Episodes)', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(fontsize=9, loc='upper left')

    plt.subplot(1, 2, 2)
    for i, (clip_val, results) in enumerate(all_results.items()):
        rewards, _, _, _, _, _, _, _, _ = results
        line_style = line_styles[i % len(line_styles)]

        start_idx = max(0, len(rewards) - 50)
        episode_range = range(start_idx, len(rewards))
        reward_values = rewards[start_idx:]

        plt.plot(episode_range, reward_values,
                 label=f'ε={clip_val}',
                 color=colors[i % len(colors)],
                 linestyle=line_style,
                 linewidth=1.5,
                 alpha=0.8)

    plt.title('Zoom: Last 50 Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(fontsize=9, loc='upper left')

    plt.suptitle('Reward Convergence Comparison - Raw Values (No Moving Average)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clip_epsilon_raw_rewards_zoomed.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*100}")
    print("SUMMARY RESULTS FOR DIFFERENT CLIP_EPSILON VALUES")
    print(f"{'='*100}")
    print(f"{'Clip ε':<10} {'Final Avg Reward':<18} {'Min Reward':<15} {'Max Reward':<15} {'Avg DAGs':<12} {'Avg Violations':<15}")
    print(f"{'-'*100}")

    for clip_val, results in all_results.items():
        rewards, dags_completed, violations, backup_rates, response_times, load_deviations, reliability_scores, cloud_usage_rates, priority_rewards = results

        if len(rewards) >= 20:
            final_avg_reward = np.mean(rewards[-20:])
            final_avg_dags = np.mean(dags_completed[-20:])
            final_avg_violations = np.mean(violations[-20:])
        else:
            final_avg_reward = np.mean(rewards)
            final_avg_dags = np.mean(dags_completed)
            final_avg_violations = np.mean(violations)

        min_reward = np.min(rewards)
        max_reward = np.max(rewards)

        if clip_val == best_clip[0]:
            print(f"{clip_val:<10.2f} \033[1;32m{final_avg_reward:<18.2f}\033[0m {min_reward:<15.2f} {max_reward:<15.2f} {final_avg_dags:<12.2f} {final_avg_violations:<15.2f}")
        else:
            print(f"{clip_val:<10.2f} {final_avg_reward:<18.2f} {min_reward:<15.2f} {max_reward:<15.2f} {final_avg_dags:<12.2f} {final_avg_violations:<15.2f}")

    print(f"\n{'='*100}")
    print("REWARD VARIABILITY ANALYSIS")
    print(f"{'='*100}")
    print(f"{'Clip ε':<10} {'Std Dev':<15} {'CV (%)':<15} {'Q1':<15} {'Median':<15} {'Q3':<15}")
    print(f"{'-'*100}")

    for clip_val, results in all_results.items():
        rewards, _, _, _, _, _, _, _, _ = results

        std_dev = np.std(rewards)
        mean_val = np.mean(rewards)
        cv = (std_dev / abs(mean_val)) * 100 if mean_val != 0 else 0
        q1 = np.percentile(rewards, 25)
        median = np.percentile(rewards, 50)
        q3 = np.percentile(rewards, 75)

        if clip_val == best_clip[0]:
            print(f"{clip_val:<10.2f} \033[1;32m{std_dev:<15.2f}\033[0m {cv:<15.1f} {q1:<15.2f} {median:<15.2f} {q3:<15.2f}")
        else:
            print(f"{clip_val:<10.2f} {std_dev:<15.2f} {cv:<15.1f} {q1:<15.2f} {median:<15.2f} {q3:<15.2f}")

  
    print(f"\n{'='*100}")
    print("CONCLUSION")
    print(f"{'='*100}")
    print(f"• Best performing clip_epsilon value: \033[1;32mε={best_clip[0]}\033[0m")
    print(f"• Average reward for best value: \033[1;32m{best_clip[1]:.2f}\033[0m")

 
    print(f"\nCONVERGENCE PATTERNS ANALYSIS:")
    print(f"{'-'*50}")

    for clip_val, results in all_results.items():
        rewards, _, _, _, _, _, _, _, _ = results

        half_point = len(rewards) // 2
        first_half_avg = np.mean(rewards[:half_point])
        second_half_avg = np.mean(rewards[half_point:])
        improvement = ((second_half_avg - first_half_avg) / abs(first_half_avg)) * 100 if first_half_avg != 0 else 0

        trend = "Improving" if improvement > 0 else "Declining" if improvement < 0 else "Stable"

        print(f"• ε={clip_val}: {trend} ({improvement:+.1f}% change)")

    
    print(f"\nOptimal configuration: ε={best_clip[0]}, γ=0.9")
    print(f"Expected performance: Average reward ≈ {best_clip[1]:.2f}")
    print(f"{'='*100}")

    print(f"\nSaving raw reward data to files...")
    for clip_val, results in all_results.items():
        rewards, _, _, _, _, _, _, _, _ = results
        filename = f"rewards_clip_{clip_val}.txt"
        np.savetxt(filename, rewards, fmt='%.4f')
        print(f"  • Saved {len(rewards)} reward values for ε={clip_val} to {filename}")
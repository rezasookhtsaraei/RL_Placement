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
import json
import simpy

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

class SimpyDAGArrival:
    def __init__(self, num_dags=100, arrival_rate=0.5):
        self.num_dags = num_dags
        self.arrival_rate = arrival_rate
        self.arrival_times = []

    def generate_arrivals(self):
        env = simpy.Environment()

        def dag_arrival_process(env, arrival_rate):
            for i in range(self.num_dags):
                if i == 0:
                    yield env.timeout(0)
                else:
                    inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
                    yield env.timeout(inter_arrival_time)

                arrival_time = env.now
                self.arrival_times.append(arrival_time)

        env.process(dag_arrival_process(env, self.arrival_rate))

        env.run()

        return self.arrival_times

class DiabetesDAGEnv:
    def __init__(self, num_dags=100, phase=1, existing_dags=None, max_dags_to_process=None):
        self.num_tasks = 8
        self.num_nodes = 52
        self.phase = phase

        if existing_dags is not None:
            self.dags = existing_dags['dags']
            self.dag_priorities = existing_dags['priorities']
            self.dag_deadlines = existing_dags['deadlines']
            self.dag_arrival_times = existing_dags['arrival_times']
            self.num_dags = len(self.dags)
            self.node_hierarchy = existing_dags['node_hierarchy']
            self.delay_matrix = existing_dags['delay_matrix']
            self.bandwidth_matrix = existing_dags['bandwidth_matrix']
        else:
            self.node_hierarchy = self._create_hierarchical_structure()
            self.delay_matrix, self.bandwidth_matrix = self._create_network_matrices()

            self.dags = []
            self.dag_priorities = []
            self.dag_deadlines = []
            self.dag_arrival_times = []
            self.num_dags = num_dags

            print(f"Generating DAG arrival:")
            arrival_simulator = SimpyDAGArrival(num_dags=num_dags, arrival_rate=0.5)
            arrival_times = arrival_simulator.generate_arrivals()

            for i in range(num_dags):
                dag, priority, deadline = self._generate_random_dag_with_priority_and_deadline()
                self.dags.append(dag)
                self.dag_priorities.append(priority)
                self.dag_deadlines.append(deadline)
                self.dag_arrival_times.append(arrival_times[i])

        if max_dags_to_process is not None:
            self.max_dags_to_process = min(max_dags_to_process, self.num_dags)
        else:
            self.max_dags_to_process = self.num_dags

        self.original_arrival_times = self.dag_arrival_times.copy()
        self.current_dag_idx = 0
        self.last_completed_dag_info = None
        self.debug_backup_creation = False
        self.episode_allocation_history = []
        self.episode_timing_history = []
        self.deadline_violations_by_priority = {1: 0, 2: 0, 3: 0}
        self.reset()

    def get_environment_data(self):
        return {
            'dags': self.dags,
            'priorities': self.dag_priorities,
            'deadlines': self.dag_deadlines,
            'arrival_times': self.dag_arrival_times,
            'node_hierarchy': self.node_hierarchy,
            'delay_matrix': self.delay_matrix,
            'bandwidth_matrix': self.bandwidth_matrix
        }

    def limit_to_first_n_dags(self, n):
        self.max_dags_to_process = min(n, self.num_dags)
        print(f"Environment limited to first {self.max_dags_to_process} DAGs (out of {self.num_dags})")

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
                else:
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

        priority =np.random.choice([1, 2, 3])

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
        dag_execution_time = self.dag_absolute_completion_time - self.dag_start_time
        if dag_execution_time <= self.current_deadline:
            if self.current_priority == 1:
                return 3.0 * (1.0 - dag_execution_time / self.current_deadline)
            elif self.current_priority == 2:
                return 2.0 * (1.0 - dag_execution_time / self.current_deadline)
            else:
                return 1.0 * (1.0 - dag_execution_time / self.current_deadline)
        else:
            time_excess = dag_execution_time - self.current_deadline
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
        self.last_task_completion_time = self.dag_start_time  

        self._update_task_states()

        return self._get_obs()

    def _load_next_dag(self):
        self.current_dag_idx += 1
        if self.current_dag_idx < self.max_dags_to_process:
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
            self.last_task_completion_time = self.dag_start_time

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
            self.current_dag_idx / self.max_dags_to_process,
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

    def _update_running_tasks(self):
        if not self.task_status['running'] and not self.task_status['backup']:
            self.current_time += 0.01
            return

        next_completion_time = float('inf')

        for task_id, (_, completion_time) in self.task_status['running'].items():
            if completion_time < next_completion_time:
                next_completion_time = completion_time

        for task_id, (_, completion_time) in self.task_status['backup'].items():
            if completion_time < next_completion_time:
                next_completion_time = completion_time

        if next_completion_time < float('inf'):
            self.current_time = max(self.current_time, next_completion_time)
        else:
            self.current_time += 0.01

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

                self.last_task_completion_time = max(self.last_task_completion_time, self.current_time)
                self.makespan = self.last_task_completion_time - self.dag_start_time

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

    def step(self, action):
        reward = 0.0
        done = False
        dag_completed = False

        ld_before = self._calculate_load_deviation()

        avg_reliability_before = self._calculate_average_reliability()

        if not self.task_status['ready']:
            reward = -0.1
            self._update_running_tasks()
            ld_after = ld_before
            avg_reliability_after = avg_reliability_before
        else:
            task_id = next(iter(self.task_status['ready']))

            if action == self.num_nodes:
                reward = -0.5
                self._update_running_tasks()
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

                    self._update_running_tasks()

                else:
                    reward = -1.0
                    self._update_running_tasks()
                    ld_after = ld_before
                    avg_reliability_after = avg_reliability_before

        self._update_task_states()

        if len(self.task_status['completed']) == self.num_tasks:
            dag_completed = True

            self.dag_absolute_completion_time = self.current_time

            dag_execution_time = self.dag_absolute_completion_time - self.dag_start_time

            self.makespan = self.last_task_completion_time - self.dag_start_time

            timing_info = {
                'dag_idx': self.current_dag_idx,
                'priority': self.current_priority,
                'arrival_time': self.current_arrival_time,
                'start_time': self.dag_start_time,
                'completion_time': self.dag_absolute_completion_time,
                'execution_time': dag_execution_time,
                'makespan': self.makespan,
                'deadline': self.current_deadline,
                'deadline_met': dag_execution_time <= self.current_deadline
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

    def validate_timing(self):
        if not self.episode_timing_history:
            return {"status": "No timing data available"}

        validation_results = {
            'time_monotonic': True,
            'deadline_consistency': True,
            'makespan_consistency': True,
            'execution_time_positive': True,
            'total_dags': len(self.episode_timing_history)
        }

        for i in range(len(self.episode_timing_history) - 1):
            dag1 = self.episode_timing_history[i]
            dag2 = self.episode_timing_history[i + 1]

            if dag1['completion_time'] > dag2['start_time']:
                validation_results['time_monotonic'] = False
                print(f"Warning: DAG {dag1['dag_idx']} completed at {dag1['completion_time']:.2f} "
                      f"after DAG {dag2['dag_idx']} started at {dag2['start_time']:.2f}")

        for dag in self.episode_timing_history:
            if dag['execution_time'] < 0:
                validation_results['execution_time_positive'] = False

            if dag['makespan'] > dag['execution_time']:
                validation_results['makespan_consistency'] = False
                print(f"Warning: DAG {dag['dag_idx']} makespan ({dag['makespan']:.2f}) > "
                      f"execution_time ({dag['execution_time']:.2f})")

            if dag['deadline'] <= 0:
                validation_results['deadline_consistency'] = False

        return validation_results

    def render(self):
        priority_names = {1: "High", 2: "Medium", 3: "Low"}

        print(f"\n{'='*80}")
        print(f"Diabetes Management System - Hierarchical Infrastructure")
        print(f"{'='*80}")
        print(f"DAG {self.current_dag_idx + 1}/{self.max_dags_to_process} (Total DAGs: {self.num_dags})")
        print(f"Priority: {priority_names[self.current_priority]}")
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
            print(f"Deadline status:  On time ({time_since_arrival:.2f}/{self.current_deadline:.2f}, {time_since_arrival/self.current_deadline*100:.1f}% elapsed)")
        else:
            print(f"Deadline status:  Violated ({time_since_arrival - self.current_deadline:.2f} time units overdue)")

        if self.episode_timing_history:
            last_dag = self.episode_timing_history[-1]
            print(f"\nTiming Validation:")
            print(f"  DAG Execution Time: {last_dag['execution_time']:.2f}")
            print(f"  DAG Makespan: {last_dag['makespan']:.2f}")
            print(f"  Deadline Met: {'Yes' if last_dag['deadline_met'] else 'No'}")

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-3, gamma=0.9, clip_epsilon=0.1, entropy_coef=0.1):
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

    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

def train_ppo(env, agent, num_episodes=100, phase_name="Training", transfer_learning=False):
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

    if transfer_learning:
        for param_group in agent.policy_optimizer.param_groups:
            param_group['lr'] = 0.0001
        for param_group in agent.value_optimizer.param_groups:
            param_group['lr'] = 0.0003

    print(f"\n{'='*80}")
    print(f"Starting {phase_name} Phase")
    print(f"Total DAGs: {env.num_dags}, DAGs to process: {env.max_dags_to_process}")
    print(f"{'='*80}")

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

        if dags_completed > 0:
            response_times.append(total_response_time / dags_completed)
            load_deviations.append(total_load_deviation / dags_completed)
            reliability_scores.append(total_reliability_score / dags_completed)
            cloud_usage_rates.append(total_cloud_usage / dags_completed)
        else:
            response_times.append(0)
            load_deviations.append(0)
            reliability_scores.append(0)
            cloud_usage_rates.append(0)

        recent_rewards.append(total_reward)

        if episode == num_episodes - 1:
            total_backup_by_priority = {1: 0, 2: 0, 3: 0}
            total_dags_by_priority = {1: 0, 2: 0, 3: 0}
            total_deadline_violations_by_priority = {1: 0, 2: 0, 3: 0}

            for dag_info in env.episode_allocation_history:
                priority = dag_info['priority']
                total_dags_by_priority[priority] += 1
                total_backup_by_priority[priority] += dag_info['num_tasks_with_backup']

                if not dag_info['deadline_met']:
                    total_deadline_violations_by_priority[priority] += 1

            total_reliability = 0.0
            reliability_count = 0
            for dag_info in env.episode_allocation_history:
                for task_detail in dag_info['task_allocations_details']:
                    if task_detail['primary_node'] is not None:
                        node_id = task_detail['primary_node']
                        total_reliability += env.node_reliabilities[node_id]
                        reliability_count += 1
            avg_reliability_overall = total_reliability / reliability_count if reliability_count > 0 else 0

            total_load_deviation_sum = 0.0
            for dag_info in env.episode_allocation_history:
                total_load_deviation_sum += dag_info['avg_load_deviation']
            total_dags_processed = len(env.episode_allocation_history)
            avg_load_deviation = total_load_deviation_sum / total_dags_processed if total_dags_processed > 0 else 0

            fault_tolerance_stats = {
                'failed_dags_by_priority': {1: 0, 2: 0, 3: 0},
                'deadline_violated_dags_by_priority': {1: 0, 2: 0, 3: 0},
                'total_failed_dags': 0,
                'total_deadline_violated_dags': 0
            }

            for dag_info in env.episode_allocation_history:
                priority = dag_info['priority']
                if priority == 3:
                    continue

                dag_failed = False
                dag_deadline_violated = False

                for task_detail in dag_info['task_allocations_details']:
                    task_id = task_detail['task_id']
                    primary_node = task_detail['primary_node']
                    has_backup = task_detail['has_backup']

                    if primary_node is not None:
                        node_reliability = env.node_reliabilities[primary_node]

                        if (node_reliability < 0.5 and random.random() < 0.1) or (0.5<=node_reliability<=1 and random.random() < 0.005):
                            if (not has_backup) or (has_backup and random.random() < 0.1):
                                dag_failed = True
                                dag_deadline_violated = True

                if dag_failed:
                    fault_tolerance_stats['failed_dags_by_priority'][priority] += 1
                    fault_tolerance_stats['total_failed_dags'] += 1
                if dag_deadline_violated:
                    fault_tolerance_stats['deadline_violated_dags_by_priority'][priority] += 1
                    fault_tolerance_stats['total_deadline_violated_dags'] += 1

            timing_validation = env.validate_timing()

            print(f"\n{'='*80}")
            print(f"FINAL {phase_name.upper()} PHASE REPORT")
            print(f"{'='*80}")

            print(f"1. Total Episodes: {num_episodes}")
            print(f"2. Average Episode Reward: {np.mean(episode_rewards):.2f}")
            print(f"3. Total DAGs Processed: {len(env.episode_allocation_history)}")

            print(f"\n4. Timing Validation Results:")
            print(f"   Time Monotonic: {'PASS' if timing_validation['time_monotonic'] else 'FAIL'}")
            print(f"   Execution Time Positive: {'PASS' if timing_validation['execution_time_positive'] else 'FAIL'}")
            print(f"   Makespan Consistency: {'PASS' if timing_validation['makespan_consistency'] else 'FAIL'}")

            print(f"\n5. Deadline Violation Statistics by Priority:")
            for priority in [1, 2, 3]:
                priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
                deadline_violations_count = total_deadline_violations_by_priority[priority]
                total_dags = total_dags_by_priority[priority]
                if total_dags > 0:
                    violation_rate = deadline_violations_count / total_dags * 100
                    print(f"   {priority_name} Priority: {deadline_violations_count}/{total_dags} DAGs violated deadline ({violation_rate:.1f}%)")

            print(f"\n6. Fault Tolerance Statistics (Failed DAGs):")
            for priority in [1, 2, 3]:
                priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
                failed_dags = fault_tolerance_stats['failed_dags_by_priority'][priority]
                total_dags = total_dags_by_priority[priority]
                if total_dags > 0:
                    failure_rate = failed_dags / total_dags * 100
                    print(f"   {priority_name} Priority: {failed_dags}/{total_dags} DAGs failed ({failure_rate:.1f}%)")

            print(f"\n7. Combined Statistics (Deadline Violations + Failed DAGs):")
            total_combined = 0
            for priority in [1, 2, 3]:
                priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
                deadline_violations_count = total_deadline_violations_by_priority[priority]
                failed_dags = fault_tolerance_stats['failed_dags_by_priority'][priority]
                total_dags = total_dags_by_priority[priority]
                combined = deadline_violations_count + failed_dags
                total_combined += combined
                if total_dags > 0:
                    combined_rate = combined / total_dags * 100
                    print(f"   {priority_name} Priority: {combined}/{total_dags} DAGs with issues ({combined_rate:.1f}%)")

            print(f"\n8. Average Selected Host Reliability: {avg_reliability_overall:.4f}")
            print(f"   (Higher is better, range: 0.0-1.0)")

            print(f"\n9. Load Balance (Average Load Deviation): {avg_load_deviation:.4f}")
            print(f"   (Lower value indicates better load balance)")

            print(f"\n10. Backup Statistics by Priority:")
            for priority in [1, 2, 3]:
                priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
                total_dags = total_dags_by_priority[priority]
                total_backup = total_backup_by_priority[priority]
                if total_dags > 0:
                    avg_backup_per_dag = total_backup / total_dags
                    print(f"   {priority_name} Priority: {total_backup} total backups across {total_dags} DAGs ({avg_backup_per_dag:.2f} backups/DAG)")

            print(f"\n11. Performance Metrics Summary:")
            if len(response_times) > 0:
                print(f"   Average Response Time (F1): {np.mean(response_times):.2f}")
            if len(load_deviations) > 0:
                print(f"   Average Load Deviation (F2): {np.mean(load_deviations):.4f}")
            if len(reliability_scores) > 0:
                print(f"   Average Reliability Score (F3): {np.mean(reliability_scores):.4f}")

            print(f"\n12. Cloud Usage Statistics:")
            if len(cloud_usage_rates) > 0:
                avg_cloud_usage = np.mean(cloud_usage_rates) * 100
                print(f"   Average Cloud Usage: {avg_cloud_usage:.1f}%")

            print(f"{'='*80}")


            comprehensive_results = {
                'phase': phase_name,
                'total_episodes': num_episodes,
                'total_dags': env.num_dags,
                'dags_processed': env.max_dags_to_process,
                'avg_episode_reward': float(np.mean(episode_rewards)),
                'timing_validation': timing_validation,
                'deadline_violations_by_priority': {
                    'high': {'violations': total_deadline_violations_by_priority[1],
                            'total_dags': total_dags_by_priority[1],
                            'rate': total_deadline_violations_by_priority[1]/total_dags_by_priority[1] if total_dags_by_priority[1] > 0 else 0},
                    'medium': {'violations': total_deadline_violations_by_priority[2],
                              'total_dags': total_dags_by_priority[2],
                              'rate': total_deadline_violations_by_priority[2]/total_dags_by_priority[2] if total_dags_by_priority[2] > 0 else 0},
                    'low': {'violations': total_deadline_violations_by_priority[3],
                           'total_dags': total_dags_by_priority[3],
                           'rate': total_deadline_violations_by_priority[3]/total_dags_by_priority[3] if total_dags_by_priority[3] > 0 else 0}
                },
                'failed_dags_by_priority': {
                    'high': {'failed': fault_tolerance_stats['failed_dags_by_priority'][1],
                            'total_dags': total_dags_by_priority[1],
                            'rate': fault_tolerance_stats['failed_dags_by_priority'][1]/total_dags_by_priority[1] if total_dags_by_priority[1] > 0 else 0},
                    'medium': {'failed': fault_tolerance_stats['failed_dags_by_priority'][2],
                              'total_dags': total_dags_by_priority[2],
                              'rate': fault_tolerance_stats['failed_dags_by_priority'][2]/total_dags_by_priority[2] if total_dags_by_priority[2] > 0 else 0},
                    'low': {'failed': fault_tolerance_stats['failed_dags_by_priority'][3],
                           'total_dags': total_dags_by_priority[3],
                           'rate': fault_tolerance_stats['failed_dags_by_priority'][3]/total_dags_by_priority[3] if total_dags_by_priority[3] > 0 else 0}
                },
                'combined_issues_by_priority': {
                    'high': {'issues': total_deadline_violations_by_priority[1] + fault_tolerance_stats['failed_dags_by_priority'][1],
                            'total_dags': total_dags_by_priority[1],
                            'rate': (total_deadline_violations_by_priority[1] + fault_tolerance_stats['failed_dags_by_priority'][1])/total_dags_by_priority[1] if total_dags_by_priority[1] > 0 else 0},
                    'medium': {'issues': total_deadline_violations_by_priority[2] + fault_tolerance_stats['failed_dags_by_priority'][2],
                              'total_dags': total_dags_by_priority[2],
                              'rate': (total_deadline_violations_by_priority[2] + fault_tolerance_stats['failed_dags_by_priority'][2])/total_dags_by_priority[2] if total_dags_by_priority[2] > 0 else 0},
                    'low': {'issues': total_deadline_violations_by_priority[3] + fault_tolerance_stats['failed_dags_by_priority'][3],
                           'total_dags': total_dags_by_priority[3],
                           'rate': (total_deadline_violations_by_priority[3] + fault_tolerance_stats['failed_dags_by_priority'][3])/total_dags_by_priority[3] if total_dags_by_priority[3] > 0 else 0}
                },
                'avg_host_reliability': float(avg_reliability_overall),
                'avg_load_deviation': float(avg_load_deviation),
                'avg_response_time': float(np.mean(response_times)) if len(response_times) > 0 else 0,
                'avg_reliability_score': float(np.mean(reliability_scores)) if len(reliability_scores) > 0 else 0,
                'avg_cloud_usage': float(np.mean(cloud_usage_rates)) if len(cloud_usage_rates) > 0 else 0,
                'backup_statistics': {
                    'high': {'total_backups': total_backup_by_priority[1],
                            'total_dags': total_dags_by_priority[1],
                            'avg_backups_per_dag': total_backup_by_priority[1]/total_dags_by_priority[1] if total_dags_by_priority[1] > 0 else 0},
                    'medium': {'total_backups': total_backup_by_priority[2],
                              'total_dags': total_dags_by_priority[2],
                              'avg_backups_per_dag': total_backup_by_priority[2]/total_dags_by_priority[2] if total_dags_by_priority[2] > 0 else 0},
                    'low': {'total_backups': total_backup_by_priority[3],
                           'total_dags': total_dags_by_priority[3],
                           'avg_backups_per_dag': total_backup_by_priority[3]/total_dags_by_priority[3] if total_dags_by_priority[3] > 0 else 0}
                }
            }

            return comprehensive_results

        elif episode % 10 == 0:
            print(f"\n{phase_name} - Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.2f}, DAGs Completed: {dags_completed}/{env.max_dags_to_process}")

    return None

if __name__ == "__main__":

    #SEED = x
    #random.seed(SEED)
    #np.random.seed(SEED)
    #torch.manual_seed(SEED)


    TOTAL_DAGS = 2700
    PHASE1_DAGS = 900

    print(f"\n{'='*80}")
    print(f"Diabetes Management System - Hierarchical Infrastructure")
    print(f"{'='*80}")
    print(f"Using SimPy for DAG arrival simulation")
    print(f"Infrastructure: 1 Cloud + 50 Fog (25 Fog2 + 25 Fog1) + 1 Edge = 52 nodes")
    print(f"Total DAGs: {TOTAL_DAGS}")
    print(f"Phase 1 DAGs: {PHASE1_DAGS}")
    print(f"{'='*80}\n")


    print(f"Generating all {TOTAL_DAGS} DAGs using SimPy for arrival simulation...")
    all_dags_env = DiabetesDAGEnv(num_dags=TOTAL_DAGS)
    all_dags_data = all_dags_env.get_environment_data()


    print("\n" + "="*80)
    print(f"PHASE 1: Training on first {PHASE1_DAGS} DAGs")
    print("="*80)


    env_phase1 = DiabetesDAGEnv(num_dags=TOTAL_DAGS, phase=1, existing_dags=all_dags_data, max_dags_to_process=PHASE1_DAGS)
    state_dim = env_phase1.num_nodes * 3 + env_phase1.num_nodes * 3 + env_phase1.num_tasks + 3 + 5 + 2
    action_dim = env_phase1.num_nodes + 1


    agent = PPO(state_dim, action_dim, lr=3e-4, entropy_coef=0.1)


    start_time_phase1 = time.time()
    phase1_results = train_ppo(env_phase1, agent, num_episodes=100, phase_name="Phase 1 ")
    end_time_phase1 = time.time()

    print(f"\nPhase 1 completed in {end_time_phase1 - start_time_phase1:.2f} seconds")

    agent.save_model("phase1_model.pth")


    print("\n" + "="*80)
    print("PHASE 2: Transfer Learning on ALL DAGs")
    print("="*80)


    env_phase2 = DiabetesDAGEnv(num_dags=TOTAL_DAGS, phase=2, existing_dags=all_dags_data, max_dags_to_process=TOTAL_DAGS)


    agent_phase2 = PPO(state_dim, action_dim, lr=3e-4, entropy_coef=0.1)


    agent_phase2.load_model("phase1_model.pth")


    start_time_phase2 = time.time()
    phase2_results = train_ppo(env_phase2, agent_phase2, num_episodes=100,
                               phase_name="Phase 2 (All DAGs with Transfer Learning)",
                               transfer_learning=True)
    end_time_phase2 = time.time()

    print(f"\nPhase 2 completed in {end_time_phase2 - start_time_phase2:.2f} seconds")


    agent_phase2.save_model("final_model.pth")


    print("\n" + "="*80)
    print("COMPARISON SUMMARY: Phase 1 vs Phase 2")
    print("="*80)

    if phase1_results and phase2_results:
        print(f"\n1. DAG Processing Comparison:")
        print(f"   Phase 1: Processed {phase1_results['dags_processed']} DAGs")
        print(f"   Phase 2: Processed {phase2_results['dags_processed']} DAGs")

        print(f"\n2. Performance Improvement:")
        print(f"   Phase 1 Average Reward: {phase1_results['avg_episode_reward']:.2f}")
        print(f"   Phase 2 Average Reward: {phase2_results['avg_episode_reward']:.2f}")

        if phase1_results['avg_episode_reward'] != 0:
            improvement = ((phase2_results['avg_episode_reward'] - phase1_results['avg_episode_reward']) /
                          abs(phase1_results['avg_episode_reward'])) * 100
            print(f"   Improvement: {improvement:.1f}%")

        print(f"\n3. Timing Validation Comparison:")
        print(f"   Phase 1 Time Monotonic: {'PASS' if phase1_results['timing_validation']['time_monotonic'] else 'FAIL'}")
        print(f"   Phase 2 Time Monotonic: {'PASS' if phase2_results['timing_validation']['time_monotonic'] else 'FAIL'}")

        print(f"\n4. Reliability Comparison:")
        print(f"   Phase 1 Avg Host Reliability: {phase1_results['avg_host_reliability']:.4f}")
        print(f"   Phase 2 Avg Host Reliability: {phase2_results['avg_host_reliability']:.4f}")

        print(f"\n5. Load Balance Comparison:")
        print(f"   Phase 1 Avg Load Deviation: {phase1_results['avg_load_deviation']:.4f}")
        print(f"   Phase 2 Avg Load Deviation: {phase2_results['avg_load_deviation']:.4f}")

        print(f"\n6. Combined Issues (Deadline Violations + Failed DAGs):")
        for priority_name in ['high', 'medium', 'low']:
            phase1_rate = phase1_results['combined_issues_by_priority'][priority_name]['rate'] * 100
            phase2_rate = phase2_results['combined_issues_by_priority'][priority_name]['rate'] * 100
            improvement = phase1_rate - phase2_rate
            print(f"   {priority_name.capitalize()} Priority: {phase1_rate:.1f}% -> {phase2_rate:.1f}% (Improvement: {improvement:.1f}%)")

    print(f"\n{'='*80}")
    print(f"Simulation Complete")
    print(f"Total Training Time: {(end_time_phase2 - start_time_phase1):.2f} seconds")
    print(f"{'='*80}")


    results = {
        'phase1': phase1_results,
        'phase2': phase2_results,
        'training_time': {
            'phase1': end_time_phase1 - start_time_phase1,
            'phase2': end_time_phase2 - start_time_phase2,
            'total': end_time_phase2 - start_time_phase1
        }
    }

    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to 'training_results.json'")
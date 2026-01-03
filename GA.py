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
import simpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PRIORITY_WEIGHTS = {
    1: {
        'response_time': 0.5,
        'load_balance': 0.5
    },
    2: {
        'response_time': 0.5,
        'load_balance': 0.5
    },
    3: {
        'response_time': 0.5,
        'load_balance': 0.5
    }
}

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

        self.simpy_env = simpy.Environment()

        self.dag_queue = simpy.Store(self.simpy_env)

        self.dags = []
        self.dag_priorities = []
        self.dag_deadlines = []
        self.arrival_rate = 0.5

        self.simpy_env.process(self._dag_generator())

        self.simpy_env.run()

        self.original_arrival_times = self.dag_arrival_times.copy()
        self.current_dag_idx = 0
        self.last_completed_dag_info = None
        self.episode_allocation_history = []
        self.episode_timing_history = []
        self.deadline_violations_by_priority = {1: 0, 2: 0, 3: 0}
        self.reset()

    def _dag_generator(self):
        for i in range(self.num_dags):
            dag, priority, deadline = self._generate_random_dag_with_priority_and_deadline()

            if i == 0:
                arrival_time = 0.0
            else:
                arrival_time = np.random.exponential(1.0/self.arrival_rate)

            yield self.simpy_env.timeout(arrival_time)

            self.dags.append(dag)
            self.dag_priorities.append(priority)
            self.dag_deadlines.append(deadline)
            self.dag_arrival_times.append(float(self.simpy_env.now))

            yield self.dag_queue.put({
                'dag': dag,
                'priority': priority,
                'deadline': deadline,
                'arrival_time': float(self.simpy_env.now),
                'index': i
            })

            print(f"DAG {i} generated at time {self.simpy_env.now:.2f} with priority {priority}")

        print(f"Total {self.num_dags} DAGs generated")

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
        for queued_task in self.node_queues[node_id]:
            queue_load += self.dag.nodes[queued_task]['processing_need']

        current_processing_need = self.dag.nodes[task_id]['processing_need']
        computing_power = self.node_hierarchy['computing_power'][node_id]

        pt = (queue_load + current_processing_need) / computing_power

        response_time = rdtt + pt

        return response_time

    def _calculate_execution_time(self, task_id, node_id):
        processing_need = self.dag.nodes[task_id]['processing_need']
        computing_power = self.node_hierarchy['computing_power'][node_id]

        base_time = processing_need / computing_power

        queue_load = 0.0
        for task in self.node_queues[node_id]:
            queue_load += self.dag.nodes[task]['processing_need']

        execution_time = base_time + (queue_load / computing_power)

        return execution_time

    def _calculate_load_deviation(self):
        cpu_demands = np.zeros(self.num_nodes, dtype=np.float32)
        ram_demands = np.zeros(self.num_nodes, dtype=np.float32)

        for node_id in range(self.num_nodes):
            for task_id in self.node_queues[node_id]:
                cpu_demands[node_id] += self.dag.nodes[task_id]['processing_need']
                ram_demands[node_id] += self.dag.nodes[task_id]['ram_need']

            for task_id, (running_node, _) in self.task_status['running'].items():
                if running_node == node_id:
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
        if self.dag_absolute_completion_time is not None:
            dag_execution_time = self.dag_absolute_completion_time - self.dag_start_time
        else:
            dag_execution_time = self.makespan

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
        self.load_deviation_history = []
        self.last_task_completion_time = 0.0

        self.node_ram_capacities = self.node_hierarchy['ram_capacity'].copy()
        self.node_storage_capacities = self.node_hierarchy['storage_capacity'].copy()
        self.node_computing_powers = self.node_hierarchy['computing_power'].copy()

        self.remaining_ram = self.node_ram_capacities.copy()
        self.remaining_storage = self.node_storage_capacities.copy()
        self.node_reliabilities = self.node_hierarchy['reliability'].copy()

        self.node_queues = [deque() for _ in range(self.num_nodes)]

        self.task_status = {
            'waiting': set(range(self.num_tasks)),
            'ready': set(),
            'running': {},
            'completed': set(),
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
        if self.current_dag_idx < self.num_dags:
            self.dag = copy.deepcopy(self.dags[self.current_dag_idx])
            self.current_priority = self.dag_priorities[self.current_dag_idx]
            self.current_deadline = self.dag_deadlines[self.current_dag_idx]
            self.current_arrival_time = self.dag_arrival_times[self.current_dag_idx]

            self.current_time = max(self.current_time, self.current_arrival_time)

            self.remaining_ram = self.node_ram_capacities.copy()
            self.remaining_storage = self.node_storage_capacities.copy()

            self.node_queues = [deque() for _ in range(self.num_nodes)]

            self.task_status = {
                'waiting': set(range(self.num_tasks)),
                'ready': set(),
                'running': {},
                'completed': set(),
                'primary_completion_times': {}
            }

            self.task_allocations = {}

            self.makespan = 0
            self.dag_start_time = self.current_time
            self.total_response_time = 0.0
            self.total_load_deviation = 0.0
            self.load_deviation_history = []
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

        queue_status = np.zeros(self.num_nodes, dtype=np.float32)
        for node_id in range(self.num_nodes):
            queue_status[node_id] = len(self.node_queues[node_id])

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

        if not self.task_status['ready']:
            reward = -0.1
            self._update_running_tasks_improved()
            ld_after = ld_before
        else:
            task_id = next(iter(self.task_status['ready']))

            if action == self.num_nodes:
                reward = -0.5
                ld_after = ld_before
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

                    self.node_queues[action].append(task_id)

                    execution_time = self._calculate_execution_time(task_id, action)

                    actual_start_time = self.current_time + execution_time
                    self.task_status['running'][task_id] = (action, actual_start_time)
                    self.task_allocations[task_id] = action

                    ld_after = self._calculate_load_deviation()

                    load_balance_reward = self._calculate_load_balance_reward(ld_before, ld_after)

                    cloud_penalty = self._calculate_cloud_overuse_penalty(action)
                    diversity_reward = self._calculate_diversity_reward(action)
                    capacity_reward = self._calculate_capacity_reward(action, ram_need, storage_need)

                    weights = PRIORITY_WEIGHTS[self.current_priority]
                    w1, w2 = weights['response_time'], weights['load_balance']

                    scaled_load_balance_reward = load_balance_reward * 8.0

                    combined_reward = (
                        w1 * response_reward +
                        w2 * scaled_load_balance_reward
                    )

                    reward = combined_reward
                    self.total_load_deviation += ld_after
                    self.load_deviation_history.append(ld_after)

                    self._update_running_tasks_improved()

                else:
                    reward = -1.0
                    ld_after = ld_before
                    self._update_running_tasks_improved()

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

            weights = PRIORITY_WEIGHTS[self.current_priority]

            f1_reward = -self.total_response_time * 2.0
            f2_reward = -avg_load_deviation * 10.0
            deadline_reward = self._calculate_deadline_reward()

            w1_final = weights['response_time'] * 0.5
            w2_final = weights['load_balance'] * 0.5

            final_reward = (
                w1_final * f1_reward +
                w2_final * f2_reward
            )


            reward += final_reward
            self.dag_rewards.append(final_reward)

            self.last_completed_dag_info = {
                'dag_idx': self.current_dag_idx,
                'priority': self.current_priority,
                'weights': weights,
                'makespan': self.makespan,
                'deadline': self.current_deadline,
                'deadline_met': dag_execution_time <= self.current_deadline,
                'deadline_reward': deadline_reward,
                'total_response_time': self.total_response_time,
                'avg_response_time': self.total_response_time / self.num_tasks if self.num_tasks > 0 else 0,
                'avg_load_deviation': avg_load_deviation,
                'total_load_deviation': self.total_load_deviation,
                'arrival_time': self.current_arrival_time,
                'start_time': self.dag_start_time,
                'completion_time': self.dag_absolute_completion_time,
                'execution_time': dag_execution_time,
                'task_allocations': copy.deepcopy(self.task_allocations)
            }

            self.episode_allocation_history.append(self.last_completed_dag_info)

            if not self._load_next_dag():
                done = True

        self.total_reward += reward

        return self._get_obs(), reward, done, dag_completed

    def _update_running_tasks(self):
        completed_tasks = []

        for task_id, (node_id, completion_time) in self.task_status['running'].items():
            if self.current_time >= completion_time:
                completed_tasks.append(task_id)

                ram_need = self.dag.nodes[task_id]['ram_need']
                storage_need = self.dag.nodes[task_id]['storage_need']
                self.remaining_ram[node_id] += ram_need
                self.remaining_storage[node_id] += storage_need

                if task_id in self.node_queues[node_id]:
                    self.node_queues[node_id].remove(task_id)

                self.last_task_completion_time = max(self.last_task_completion_time, self.current_time)
                self.makespan = self.last_task_completion_time - self.dag_start_time
                self.task_status['primary_completion_times'][task_id] = self.current_time

        for task_id in completed_tasks:
            del self.task_status['running'][task_id]
            self.task_status['completed'].add(task_id)

        self.current_time += 0.1

    def _update_running_tasks_improved(self):
        if not self.task_status['running']:
            self.current_time += 0.01
            return

        next_completion_time = float('inf')

        for task_id, (_, completion_time) in self.task_status['running'].items():
            if completion_time < next_completion_time:
                next_completion_time = completion_time

        if next_completion_time < float('inf'):
            self.current_time = max(self.current_time, next_completion_time)
        else:
            self.current_time += 0.01

        completed_tasks = []

        for task_id, (node_id, completion_time) in self.task_status['running'].items():
            if self.current_time >= completion_time:
                completed_tasks.append(task_id)

                ram_need = self.dag.nodes[task_id]['ram_need']
                storage_need = self.dag.nodes[task_id]['storage_need']
                self.remaining_ram[node_id] += ram_need
                self.remaining_storage[node_id] += storage_need

                if task_id in self.node_queues[node_id]:
                    self.node_queues[node_id].remove(task_id)

                self.last_task_completion_time = max(self.last_task_completion_time, self.current_time)
                self.makespan = self.last_task_completion_time - self.dag_start_time
                self.task_status['primary_completion_times'][task_id] = self.current_time

        for task_id in completed_tasks:
            del self.task_status['running'][task_id]
            self.task_status['completed'].add(task_id)

    def render(self):
        priority_names = {1: "High", 2: "Medium", 3: "Low"}

        print(f"\n{'='*80}")
        print(f"Diabetes Management System - Hierarchical Infrastructure")
        print(f"{'='*80}")
        print(f"DAG {self.current_dag_idx + 1}/{self.num_dags} (Priority: {priority_names[self.current_priority]})")
        print(f"Arrival Time: {self.current_arrival_time:.2f}, Current Time: {self.current_time:.2f}")
        print(f"Deadline: {self.current_deadline:.2f}, Time Since Arrival: {self.current_time - self.current_arrival_time:.2f}")

        weights = PRIORITY_WEIGHTS[self.current_priority]
        print(f"Weights - Response: {weights['response_time']:.2f}, Load Balance: {weights['load_balance']:.2f}")

        print(f"Time: {self.current_time:.2f}, Makespan: {self.makespan:.2f}")
        print(f"Total Response Time (F1): {self.total_response_time:.2f}")
        print(f"Average Response Time per Task: {self.total_response_time/self.num_tasks:.2f}" if self.num_tasks > 0 else "No tasks")
        print(f"Total Load Deviation (F2): {self.total_load_deviation:.4f}")
        print(f"Current Load Deviation: {self._calculate_load_deviation():.4f}")

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
        print(f"Ready tasks: {len(self.task_status['ready'])}")
        print(f"Running tasks: {len(self.task_status['running'])}")

        time_since_arrival = self.current_time - self.current_arrival_time
        if time_since_arrival <= self.current_deadline:
            print(f"Deadline status:  On time ({time_since_arrival:.2f}/{self.current_deadline:.2f}, {time_since_arrival/self.current_deadline*100:.1f}% elapsed)")
        else:
            print(f"Deadline status:  Violated ({time_since_arrival - self.current_deadline:.2f} time units overdue)")

class GA:
    def __init__(self, env, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.env = env
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_tasks = env.num_tasks
        self.num_nodes = env.num_nodes
        self.population = []
        self.fitness_history = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            chromosome = [random.randint(0, self.num_nodes - 1) for _ in range(self.num_tasks)]
            self.population.append(chromosome)

    def evaluate_chromosome(self, chromosome):
        self.env.reset()
        total_reward = 0
        done = False

        while not done:
            if not self.env.task_status['ready']:
                action = self.num_nodes
            else:
                task_id = min(self.env.task_status['ready'])
                action = chromosome[task_id]

            _, reward, done, _ = self.env.step(action)
            total_reward += reward

        return total_reward

    def evaluate_chromosome_all_dags(self, chromosome):
        self.env.reset()
        total_reward = 0
        done = False

        while not done:
            if not self.env.task_status['ready']:
                action = self.num_nodes
            else:
                task_id = min(self.env.task_status['ready'])
                action = chromosome[task_id]

            _, reward, done, _ = self.env.step(action)
            total_reward += reward

        return total_reward

    def select_parents(self, fitness_values):
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            probabilities = [1.0 / len(fitness_values)] * len(fitness_values)
        else:
            probabilities = [f / total_fitness for f in fitness_values]

        parents = []
        for _ in range(2):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    parents.append(self.population[i])
                    break

        return parents

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = random.randint(1, self.num_tasks - 2)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    def mutate(self, chromosome):
        mutated = chromosome.copy()
        for i in range(self.num_tasks):
            if random.random() < self.mutation_rate:
                mutated[i] = random.randint(0, self.num_nodes - 1)
        return mutated

    def run(self, num_generations=100):
        self.initialize_population()
        best_chromosome = None
        best_fitness = float('-inf')

        for generation in range(num_generations):
            fitness_values = []
            for chromosome in self.population:
                fitness = self.evaluate_chromosome(chromosome)
                fitness_values.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_chromosome = chromosome.copy()

            self.fitness_history.append(np.mean(fitness_values))

            new_population = []

            if best_chromosome:
                new_population.append(best_chromosome.copy())

            while len(new_population) < self.population_size:
                parents = self.select_parents(fitness_values)
                child1, child2 = self.crossover(parents[0], parents[1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]

            if generation % 10 == 0:
                print(f"Generation {generation}: Average Fitness = {np.mean(fitness_values):.2f}, Best Fitness = {best_fitness:.2f}")

        return best_chromosome

def train_ga(env, num_generations=100):
    ga = GA(env, population_size=30, mutation_rate=0.1, crossover_rate=0.8)

    print(f"\n{'='*80}")
    print(f"Starting Genetic Algorithm Training")
    print(f"{'='*80}")

    start_time = time.time()

    best_chromosome = ga.run(num_generations)

    end_time = time.time()

    print(f"\nGenetic Algorithm completed in {end_time - start_time:.2f} seconds")
    print(f"Best chromosome: {best_chromosome}")

    print(f"\n{'='*80}")
    print(f"Evaluating Best Chromosome on All DAGs")
    print(f"{'='*80}")

    env.reset()

    total_reward = 0
    done = False
    dags_completed = 0

    while not done:
        if not env.task_status['ready']:
            action = env.num_nodes
        else:
            task_id = min(env.task_status['ready'])
            action = best_chromosome[task_id]

        _, reward, done, dag_completed = env.step(action)
        total_reward += reward

        if dag_completed:
            dags_completed += 1

    total_dags_by_priority = {1: 0, 2: 0, 3: 0}
    total_deadline_violations_by_priority = {1: 0, 2: 0, 3: 0}

    for dag_info in env.episode_allocation_history:
        priority = dag_info['priority']
        total_dags_by_priority[priority] += 1

        if not dag_info['deadline_met']:
            total_deadline_violations_by_priority[priority] += 1

    total_load_deviation_sum = 0.0
    for dag_info in env.episode_allocation_history:
        total_load_deviation_sum += dag_info['avg_load_deviation']
    total_dags_processed = len(env.episode_allocation_history)
    avg_load_deviation = total_load_deviation_sum / total_dags_processed if total_dags_processed > 0 else 0

    fault_stats = {
        'failed_dags_by_priority': {1: 0, 2: 0, 3: 0},
        'total_failed_dags': 0
    }

    for dag_info in env.episode_allocation_history:
        priority = dag_info['priority']
        if priority == 3:
            continue

        dag_failed = False

        for task_id in range(env.num_tasks):
            if task_id in dag_info['task_allocations']:
                node_id = dag_info['task_allocations'][task_id]
                node_reliability = env.node_hierarchy['reliability'][node_id]

                if ( random.random()<0.3 and node_reliability < 0.5) or (random.random()<0.05 and 0.5<= node_reliability <=1):
                    dag_failed = True
                    break

        if dag_failed:
            fault_stats['failed_dags_by_priority'][priority] += 1
            fault_stats['total_failed_dags'] += 1

    host_reliabilities = []
    host_nodes_count = {}

    for dag_info in env.episode_allocation_history:
        task_allocations = dag_info['task_allocations']
        for node_id in set(task_allocations.values()):
            reliability = env.node_hierarchy['reliability'][node_id]
            host_reliabilities.append(reliability)

            node_type = env.node_hierarchy['node_type'][node_id]
            host_nodes_count[node_type] = host_nodes_count.get(node_type, 0) + 1

    avg_host_reliability = np.mean(host_reliabilities) if host_reliabilities else 0.0
    std_host_reliability = np.std(host_reliabilities) if host_reliabilities else 0.0

    print(f"\n{'='*80}")
    print(f"FINAL REPORT - Genetic Algorithm")
    print(f"{'='*80}")

    print(f"1. Total Reward: {total_reward:.2f}")
    print(f"   DAGs Completed: {dags_completed}")

    print(f"\n2. Deadline Violation Statistics:")
    for priority in [1, 2, 3]:
        priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
        deadline_violations_count = total_deadline_violations_by_priority[priority]
        total_dags = total_dags_by_priority[priority]
        if total_dags > 0:
            violation_rate = (deadline_violations_count / total_dags) * 100
            print(f"   {priority_name} Priority: {deadline_violations_count}/{total_dags} DAGs violated deadline ({violation_rate:.1f}%)")

    print(f"\n3. Load Balance (Average Load Deviation): {avg_load_deviation:.4f}")
    print(f"   (Lower value indicates better load balance)")

    print(f"\n4. Fault Statistics (Without Fault Tolerance):")
    for priority in [1, 2, 3]:
        priority_name = {1: "High", 2: "Medium", 3: "Low"}[priority]
        total_dags = total_dags_by_priority[priority]
        failed = fault_stats['failed_dags_by_priority'][priority]
        if total_dags > 0:
            failure_rate = (failed / total_dags) * 100
            print(f"   {priority_name} Priority: {failed} failed DAGs out of {total_dags} DAGs ({failure_rate:.1f}%)")

    print(f"   Total Failed DAGs: {fault_stats['total_failed_dags']}")

    print(f"\n5. Response Time Statistics:")
    avg_response_time = np.mean([dag_info['avg_response_time'] for dag_info in env.episode_allocation_history]) if env.episode_allocation_history else 0
    print(f"   Average Response Time: {avg_response_time:.2f}")

    print(f"\n6. Cloud Usage Statistics:")
    cloud_usage_rates = []
    for dag_info in env.episode_allocation_history:
        cloud_tasks = sum(1 for node_id in dag_info['task_allocations'].values() if node_id == 0)
        cloud_usage_rates.append(cloud_tasks / env.num_tasks)
    avg_cloud_usage = np.mean(cloud_usage_rates) if cloud_usage_rates else 0
    print(f"   Average Cloud Usage: {avg_cloud_usage*100:.1f}%")

    print(f"\n7. Host Node Reliability Statistics:")
    print(f"   Average Host Node Reliability: {avg_host_reliability:.3f}")
    print(f"   Standard Deviation of Host Node Reliability: {std_host_reliability:.3f}")
    print(f"   Host Node Distribution by Type:")
    total_host_nodes = sum(host_nodes_count.values())
    for node_type in ['cloud', 'fog2', 'fog1', 'edge']:
        count = host_nodes_count.get(node_type, 0)
        if total_host_nodes > 0:
            percentage = (count / total_host_nodes) * 100
            print(f"     {node_type}: {count} nodes ({percentage:.1f}%)")

    print(f"\n9. Genetic Algorithm Parameters:")
    print(f"   Population Size: {ga.population_size}")
    print(f"   Mutation Rate: {ga.mutation_rate}")
    print(f"   Crossover Rate: {ga.crossover_rate}")
    print(f"   Number of Generations: {num_generations}")

    print(f"\n{'='*80}")

    return best_chromosome, total_reward

if __name__ == "__main__":

    #SEED = x
    #random.seed(SEED)
    #np.random.seed(SEED)
    #torch.manual_seed(SEED)

    num_dags = 1800
    env = DiabetesDAGEnv(num_dags=num_dags)

    print(f"\n{'='*80}")
    print(f"Diabetes Management System")

    best_chromosome, total_reward = train_ga(env, num_generations=30)

    print(f"\n{'='*80}")
    print(f"Simulation Complete")
    print(f"{'='*80}")
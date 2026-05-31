import importlib
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import kmapper as km

SEED = 42
EPOCH = 20
POP_SIZE = 20


def run_optimization(problem, output_file, algorithms=None, epoch=EPOCH, pop_size=POP_SIZE, seed=SEED):
    if algorithms is None:
        algorithms = ["PSO", "GWO"]

    results = {}
    with open(output_file, 'a') as f:
        for algo_name in algorithms:
            module = importlib.import_module(f"mealpy.swarm_based.{algo_name}")
            AlgorithmClass = getattr(module, f"Original{algo_name}")

            model = AlgorithmClass(epoch=epoch, pop_size=pop_size)
            start_time = time.time()
            g_best = model.solve(problem, seed=seed)
            total_time = time.time() - start_time

            last_update_epoch = 0
            for i in range(1, len(model.history.list_global_best_fit)):
                if model.history.list_global_best_fit[i] < model.history.list_global_best_fit[i - 1]:
                    last_update_epoch = i

            convergence_time = total_time * (last_update_epoch + 1) / epoch
            best_n = int(round(g_best.solution[0]))
            best_a = round(g_best.solution[1], 3)
            best_tsc = round(-g_best.target.fitness, 4)

            f.write(f"=== {algo_name} ===\n")
            f.write(f"Best n: {best_n}\n")
            f.write(f"Best a: {best_a}\n")
            f.write(f"Best TSC: {best_tsc}\n")
            f.write(f"Convergence epoch: {last_update_epoch}\n")
            f.write(f"Convergence time: {convergence_time:.4f}s\n")
            f.write(f"Total time: {total_time:.4f}s\n\n")
            f.flush()

            results[algo_name] = {"n": best_n, "a": best_a, "tsc": best_tsc}

    return results


def draw_graph(graph, projected_data, output_path=None, node_size=400, cmap='viridis'):
    color_list = [
        np.mean(projected_data[list(idx_list)])
        for idx_list in graph['nodes'].values()
    ]
    G = km.adapter.to_networkx(graph)
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_color=color_list,
            node_size=node_size, cmap=cmap)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    return G, color_list

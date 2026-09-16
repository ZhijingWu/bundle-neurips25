import numpy as np
import random
import argparse
import torch
from torch_geometric.utils import k_hop_subgraph, dropout_edge
import torch.nn.functional as F
import time
import os
import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from queryhelper import QueryHelper
from utils import *

class Solver:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.device}')
        self.raw_texts, self.labels = prepare_text(args.dataset)
        self.graph_data, self.init_embs = prepare_graph(args.dataset, args.encoder)
        self.feature_similarity = torch.cdist(self.init_embs, self.init_embs, p=2)
        self.query_helper = QueryHelper(args, args.dataset, args.model, self.labels)
    
    def solve(self):
        setup_seed(self.args.seed)
        original_bundles, original_bundle_classes = self.bundle_presample()
        if len(original_bundles) == 0:
            print("No valid bundles found.")
            return
        
        accs, times = [], []
        for seed in range(self.args.seed, self.args.seed + self.args.repeat):
            setup_seed(seed)
            bundles, bundle_classes = deepcopy(original_bundles), deepcopy(original_bundle_classes)
            self.model = prepare_model(self.args.gnn_type, len(self.labels), self.args.num_layers, args=self.args)
            time_start = time.time()
            for stage_idx, stage in enumerate(self.args.stages):
                self.bundle_optimize(bundles, bundle_classes, stage)
                if stage_idx < len(self.args.stages) - 1:
                    bundles, bundle_classes = self.bundle_resample(bundles, bundle_classes, requery=self.args.requery)
                    if len(bundle_classes) == 0:
                        print("No valid bundles found.")
                        break
            
            acc = self.evaluate()
            times.append(time.time() - time_start)
            accs.append(acc)
        accs = np.array(accs)
        times = np.array(times)
        print(f"Accuracy: {accs.mean():.4f} ± {accs.std():.4f}, Time: {times.mean():.4f} ± {times.std():.4f}")

    def bundle_presample(self):
        # sample bundles
        bundles = []
        for i in range(self.args.num_samples):
            bundle_nodes = self.bundle_sample(self.args.bundle_size, self.args.sample_criterion)
            bundles.append(bundle_nodes)
        
        # batched query
        bundles, bundle_classes = self.batch_bundle_query(bundles, self.args.query_type)

        if len(bundles) == 0:
            print("No valid bundles found.")
        return bundles, bundle_classes

    def bundle_resample(self, bundles, bundle_classes, requery=False):
        """
        Refine each bundle by removing the node with the lowest
        GNN confidence for the current bundle label.
        """

        print(f"[Refinement] Starting with {len(bundles)} bundles")

        self.model.eval()

        with torch.no_grad():
            logits = self.model(
                self.init_embs.to(self.device),
                self.graph_data.edge_index.to(self.device)
            )
            probs = F.softmax(logits, dim=1)

        new_bundles = []
        new_bundle_classes = []

        for bundle, bundle_class in zip(bundles, bundle_classes):
            if len(bundle) <= 1:
                continue

            bundle_tensor = torch.tensor(
                bundle,
                dtype=torch.long,
                device=self.device
            )

            # Confidence of every node for the LLM-provided bundle label.
            label_confidence = probs[bundle_tensor, bundle_class]

            # Remove the least confident node.
            remove_idx = torch.argmin(label_confidence).item()
            removed_node = bundle[remove_idx]

            # ---- Refinement diagnostics: observation only ----
            true_labels = self.graph_data.y[bundle_tensor].detach().cpu().tolist()
            confidences = label_confidence.detach().cpu().tolist()

            purity_before = sum(
                int(label == bundle_class) for label in true_labels
            ) / len(true_labels)

            remaining_true_labels = [
                label
                for i, label in enumerate(true_labels)
                if i != remove_idx
            ]

            purity_after = sum(
                int(label == bundle_class) for label in remaining_true_labels
            ) / len(remaining_true_labels)

            print(f"\n[Refinement Diagnostic]")
            print(f"LLM bundle label: {bundle_class}")

            for i, (node, true_label, confidence) in enumerate(
                zip(bundle, true_labels, confidences)
            ):
                marker = " <-- removed" if i == remove_idx else ""
                print(
                    f"node={node}  "
                    f"true={true_label}  "
                    f"confidence={confidence:.4f}"
                    f"{marker}"
                )

            print(
                f"purity: {purity_before:.4f} -> {purity_after:.4f}  "
                f"(delta={purity_after - purity_before:+.4f})"
            )

            refined_bundle = [
                node
                for i, node in enumerate(bundle)
                if i != remove_idx
            ]

            new_bundles.append(refined_bundle)
            new_bundle_classes.append(bundle_class)

        self.model.train()

        if requery:
            new_bundles, new_bundle_classes = self.batch_bundle_query(
                new_bundles,
                self.args.query_type
            )

        print(
            f"[Refinement] Bundle sizes: "
            f"{[len(b) for b in bundles]} -> "
            f"{[len(b) for b in new_bundles]}"
        )

        return new_bundles, new_bundle_classes
    
    def batch_bundle_query(self, bundles, query_type):
        def query_helper(bundle_nodes, query_type):
            bundle_class = self.bundle_query(bundle_nodes, query_type)
            if bundle_class < 0:
                return None
            return bundle_nodes, bundle_class

        new_bundles = []
        new_bundle_classes = []

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(query_helper, bundle, query_type) for bundle in bundles]
            for f in tqdm.tqdm(as_completed(futures), total=len(bundles), desc='Bundle Query'):
                result = f.result()
                if result is not None:
                    bundle_nodes, bundle_class = result
                    new_bundles.append(bundle_nodes)
                    new_bundle_classes.append(bundle_class)

        self.bundle_accuracy(new_bundles, new_bundle_classes)
        return new_bundles, new_bundle_classes

    
    def bundle_accuracy(self, bundles, bundle_classes):
        bundle_valid_rate = len(bundles) / self.args.num_samples * 100
        bundle_correct = 0
        for i, bundle in enumerate(bundles):
            bundle_class = bundle_classes[i]
            individual_classes = [self.graph_data.y[node].item() for node in bundle]
            counter = Counter(individual_classes)
            if counter.most_common(1)[0][1] == counter[bundle_class]:
                bundle_correct += 1
        bundle_class_acc = bundle_correct / len(bundles) * 100
        print(f"Bundle valid rate: {bundle_valid_rate:.4f}%, Bundle class acc: {bundle_class_acc:.4f}%")
        return bundle_valid_rate, bundle_class_acc

    def bundle_optimize(self, bundles, bundle_classes, num_epochs):
        # optimize the model
        self.model.train().to(self.device)
        self.graph_data.to(self.device)
        self.init_embs = self.init_embs.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            edge_index, _ = dropout_edge(self.graph_data.edge_index, p=self.args.edge_dropping, training=True)
            logits = self.model(self.init_embs, edge_index)
            loss = self.bundle_loss(logits, deepcopy(bundle_classes), deepcopy(bundles), self.args.loss_type)
            loss.backward()
            optimizer.step()

        return

    
    def bundle_sample(self, bundle_size, sample_criterion='neighbor'):
        # sample core
        core_node = random.choice(range(len(self.graph_data.y)))
        bundle_nodes = [core_node]

        # sample bundle
        if sample_criterion == 'neighbor':
            for hop_num in range(1, self.args.max_hop + 1):
                khop_subset, _, _, _ = k_hop_subgraph(core_node, hop_num, self.graph_data.edge_index, num_nodes=self.graph_data.num_nodes)
                khop_subset = khop_subset.tolist()
                if core_node in khop_subset:
                    khop_subset.remove(core_node)
                if len(khop_subset) >= bundle_size - 1:
                    break
            if len(khop_subset) < bundle_size - 1:
                return self.bundle_sample(bundle_size, sample_criterion)
            else:
                candidate_nodes = list(khop_subset)
                additional_nodes = random.sample(candidate_nodes, bundle_size - len(bundle_nodes))
                bundle_nodes.extend(additional_nodes)
        elif sample_criterion == 'feature':
            # find bundle_size similar nodes
            distances = self.feature_similarity[core_node]
            bundle_nodes = torch.argsort(distances)[:bundle_size].tolist()
        else:
            raise NotImplementedError(f"Sample criterion {sample_criterion} is not implemented.")
        return bundle_nodes
    
    def bundle_query(self, bundle, query_type):
        individual_labels = self.graph_data.y[bundle]
        mode_label = torch.mode(individual_labels).values.item()
        if query_type == 'gpt':
            text_list = [self.raw_texts[node] for node in bundle]
            bundle_pred = self.query_helper.query(text_list)
            return bundle_pred
        else:
            raise NotImplementedError(f"Query type {query_type} is not implemented.")
        
    def bundle_loss(self, logits, bundle_classes, bundles, loss_type):
        bundles_indices = torch.tensor(bundles, device=self.device, dtype=torch.long)  # shape: (num_samples, bundle_size)
        bundle_logits = logits[bundles_indices, :]  # shape: (num_samples, bundle_size, num_classes)
        bundle_classes = torch.tensor(bundle_classes, device=self.device, dtype=torch.long)  # shape: (num_samples, )
        num_bundles, bundle_size, num_classes = bundle_logits.shape
        if loss_type == 'average':
            average_logits = torch.mean(bundle_logits, dim=1)  # shape: (num_samples, num_classes)
            loss = F.cross_entropy(average_logits, bundle_classes)
            return loss
        elif loss_type in ['ranking']:
            bundle_prob = F.softmax(bundle_logits, dim=2)  # shape: (num_samples, bundle_size, num_classes)
            bundle_prob_mean = torch.mean(bundle_prob, dim=1)  # shape: (num_samples, num_classes)
            bundle_prob_class = bundle_prob_mean[
                torch.arange(num_bundles),
                bundle_classes
            ]  # shape: (num_samples, )
            bundle_prob_max = torch.max(bundle_prob_mean, dim=1).values  # shape: (num_samples, )
            loss = (
                -torch.clamp(
                    bundle_prob_class.log() - bundle_prob_max.log(),
                    max=0
                ).sum()
                + F.cross_entropy(
                    torch.mean(bundle_logits, dim=1),
                    bundle_classes
                )
            )
            return loss
        else:
            raise NotImplementedError(f"Loss type {loss_type} is not implemented.")

    def evaluate(self, split='test'):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.init_embs, self.graph_data.edge_index)
            pred = logits.argmax(dim=1)

            if split == 'test':
                indices = self.graph_data.test_id
            else:
                indices = self.graph_data.val_id
            acc = (pred[indices] == self.graph_data.y[indices]).float().mean().item()
        self.model.train()
        return acc

def main():
    parser = argparse.ArgumentParser(description='test the generalization of encoders')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', dest = 'device', default = 0, type = int)
    parser.add_argument('--task', dest = 'task', default = 'nc', help = 'nc refers to node classification')
    parser.add_argument('--repeat', dest = 'repeat', default = 1, type = int, help = 'repeat times')
    parser.add_argument('--dataset', dest = 'dataset', type = str, default = 'cora', help = 'cora')
    parser.add_argument('--model', dest = 'model', type = str, default = 'gpt-4o', help = 'here model refers to the LLM used to get class embeddings')
    parser.add_argument('--encoder', dest = 'encoder', type = str, default = 'llmicl_class_aware', help = 'encoders to select')
    parser.add_argument('--num_samples', type=int, default=50, help='number of samples')
    parser.add_argument('--bundle_size', type=int, default=5, help='bundle size')
    parser.add_argument('--sample_criterion', type=str, default='neighbor', help='sample criterion')
    parser.add_argument('--max_hop', type=int, default=2, help='max hop')
    parser.add_argument('--query_type', type=str, default='optimal', help='query type')
    parser.add_argument('--loss_type', type=str, default='average', help='loss type')
    parser.add_argument('--resample', action='store_true', help='resample the bundle')
    parser.add_argument('--requery', action='store_true', help='requery the bundle')
    parser.add_argument('--gnn_type', type= str, default = 'gcn', help = 'gnn type')
    parser.add_argument('--stages', type=int, nargs='+', default=[50, 50, 50], help='stages to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--edge_dropping', type=float, default=0.0)
    parser.add_argument('--expid', type=str, default='trial', help='experiment id')
    parser.add_argument('--cache_file', type=str, default=None, help='cache file')
    parser.add_argument('--disable_cache', action='store_true', help='disable cache')
    args = parser.parse_args()

    if args.cache_file is None:
        args.cache_file = f'./cache/{args.dataset}_{args.model}.pkl'
    args.output_dir = f'./output/{args.dataset}_{args.model}_{args.expid}'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    solver = Solver(args)
    solver.solve()

if __name__ == '__main__':
    main()
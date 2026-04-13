import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from flcore.clients.clientbci import *
from utils.data_utils import read_client_data
from threading import Thread
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import torch.utils.benchmark as benchmark


class FedBCI:
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_modules = copy.deepcopy(args.model)
        self.global_head_model = None
        self.global_cs_model = None
        self.global_model_copy = None
        self.hf_aggregated_model = None
        self.hf_aggregated_cs = None
        self.hf_aggregated_head = None
        self.merged_result_ori1 = None
        self.merged_results = []
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.num_clusters = args.num_clusters
        self.clients = []
        self.selected_clients = []
        self.cluster_aggregated_models = []
        self.uploaded_weights = []
        self.normalized_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.rs_test_acc = []
        self.rs_train_loss = []
        self.times = times
        self.eval_gap = args.eval_gap
        self.last_model_state = None
        self.last_accuracy = 0
        self.counter = 0
        self.cluster_aggregated_heads = []
        self.cluster_aggregated_cs = []
        self.is_pre_round = True
        self.t_num = args.num_clusters
        self.threshold = 4
        self.maxpat = 0.5
        # mnist 7 ciafr10 5 else 1
        self.k = 1
        self.dec_round = 5
        self.flag_dec = False
        self.cluster_data_sizes = []

        in_dim = list(args.model.head.parameters())[0].shape[1]
        cs = ConditionalSelection(in_dim, in_dim).to(args.device)

        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientbci(args,
                              id=i,
                              train_samples=len(train_data),
                              test_samples=len(test_data),
                              ConditionalSelection=cs)
            self.clients.append(client)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")


        self.client_clusters = []
        self.ori_client_clusters = []
        self.cluster_assignments = []
        self.merged_results_cumulative = []
        self.prev_cluster_assignments = None
        self.Budget = []
        self.head = None
        self.cs = None
        self.agg_cluster = AgglomerativeClustering(n_clusters=self.num_clusters, linkage='ward')

    def cluster_clients(self, round):

        if getattr(self, 'is_pre_round', True):

            self.client_clusters = [self.clients]
            self.num_clusters = 1

            client_ids = [client.id for client in self.clients]
            print(f"In the pre-training stage, all clients belong to one cluster: {len(client_ids)} client number - {client_ids}")

            if round == self.dec_round:
                self.is_pre_round = False
            return

        if not self.is_pre_round:
            self.num_clusters = self.t_num


        self.client_clusters = []
        client_features = []

        for client in self.clients:

            feature_vector = client.get_personalized_parameters()
            flat_feature_vector = feature_vector.flatten()
            client_features.append(flat_feature_vector)

        client_features = np.array(client_features)


        similarity_matrix = cosine_similarity(client_features)


        distance_matrix = 1 - similarity_matrix

        agg_cluster = AgglomerativeClustering(n_clusters=self.num_clusters, affinity='precomputed', linkage='average')
        try:
            self.cluster_assignments = agg_cluster.fit_predict(distance_matrix)
        except ValueError as e:
            print(f"Error during Agglomerative Clustering: {e}")
            return


        self.client_clusters = [[] for _ in range(self.num_clusters)]

        for idx, cluster_id in enumerate(self.cluster_assignments):
            self.client_clusters[cluster_id].append(self.clients[idx])

        print(f"Current number of clusters before checking small clusters: {len(self.client_clusters)}")
        for cluster_id in range(self.num_clusters):
            client_ids = [client.id for client in self.client_clusters[cluster_id]]
            print(f"cluster {cluster_id}: {len(client_ids)} client number - {client_ids}")

        if round == self.dec_round + 1:
            self.flag_dec = True
        if self.flag_dec:
            extreme_clusters = []
            non_extreme_clusters = []
            to_delete_indices = []

            for cluster_id, cluster in enumerate(self.client_clusters):
                if len(cluster) < self.threshold:
                    extreme_clusters.append(cluster)
                    to_delete_indices.append(cluster_id)
                else:
                    non_extreme_clusters.append(cluster)

            for cluster_id in sorted(to_delete_indices, reverse=True):
                del self.client_clusters[cluster_id]

            if extreme_clusters:

                centroids = []
                for cluster in non_extreme_clusters:
                    cluster_features = np.vstack(
                        [client.get_personalized_parameters().reshape(1, -1) for client in cluster])
                    centroid = np.mean(cluster_features, axis=0)
                    centroids.append(centroid.reshape(1, -1))

                extreme_client_features = [client.get_personalized_parameters().flatten() for cluster in
                                           extreme_clusters
                                           for client in cluster]
                extreme_client_features = np.vstack(extreme_client_features)


                centroids_matrix = np.vstack([centroid.flatten() for centroid in centroids])


                similarity_matrix = cosine_similarity(extreme_client_features,
                                                      centroids_matrix)
                global_client_idx = 0

                for extreme_idx, cluster in enumerate(extreme_clusters):
                    for client_idx, client in enumerate(cluster):

                        best_cluster_idx = np.argmax(similarity_matrix[global_client_idx])
                        self.client_clusters[best_cluster_idx].append(client)
                        global_client_idx += 1


                self.num_clusters = len(non_extreme_clusters)


            print(f"Current number of clusters before split detection: {len(self.client_clusters)}")
            for cluster_id in range(self.num_clusters):
                client_ids = [client.id for client in self.client_clusters[cluster_id]]
                print(f"cluster {cluster_id}: {len(client_ids)} client number - {client_ids}")

            processing_queue = list(self.client_clusters)
            final_clusters = []


            split_threshold_count = self.maxpat * self.num_clients

            while processing_queue:

                cluster = processing_queue.pop(0)

                if len(cluster) >= split_threshold_count:
                    cluster_features = np.array([c.get_personalized_parameters() for c in cluster])

                    sub_indices_list = self.bi_partitioning(cluster_features, max_clusters=2)


                    new_sub_clusters = []
                    for sub_indices in sub_indices_list:
                        sub_client_list = [cluster[i] for i in sub_indices]
                        new_sub_clusters.append(sub_client_list)

                    processing_queue.extend(new_sub_clusters)
                else:

                    final_clusters.append(cluster)

            self.client_clusters = final_clusters
            self.num_clusters = len(self.client_clusters)


            print("final cluster:")
            for cluster_id in range(self.num_clusters):
                client_ids = [client.id for client in self.client_clusters[cluster_id]]
                print(f"cluster {cluster_id}: {len(client_ids)} client number - {client_ids}")

    def bi_partitioning(self, data, max_clusters=2, threshold=0.01):
        clusters = [list(range(len(data)))]

        while len(clusters) < max_clusters:

            largest_cluster = max(clusters, key=len)
            clusters.remove(largest_cluster)


            seed1, seed2 = data[np.random.choice(largest_cluster, 2, replace=False)]
            cluster1, cluster2 = [], []

            for idx in largest_cluster:
                dist1 = np.linalg.norm(data[idx] - seed1)
                dist2 = np.linalg.norm(data[idx] - seed2)
                if dist1 < dist2:
                    cluster1.append(idx)
                else:
                    cluster2.append(idx)

            cluster1, cluster2 = np.array(cluster1), np.array(cluster2)
            centroid1, centroid2 = np.mean(data[cluster1], axis=0), np.mean(data[cluster2], axis=0)


            while True:
                new_cluster1, new_cluster2 = [], []
                for idx in largest_cluster:
                    dist1 = np.linalg.norm(data[idx] - centroid1)
                    dist2 = np.linalg.norm(data[idx] - centroid2)
                    if dist1 < dist2:
                        new_cluster1.append(idx)
                    else:
                        new_cluster2.append(idx)

                new_cluster1, new_cluster2 = np.array(new_cluster1), np.array(new_cluster2)
                new_centroid1, new_centroid2 = np.mean(data[new_cluster1], axis=0), np.mean(data[new_cluster2], axis=0)


                if (np.linalg.norm(centroid1 - new_centroid1) < threshold and
                        np.linalg.norm(centroid2 - new_centroid2) < threshold):

                    break

                centroid1, centroid2 = new_centroid1, new_centroid2
                cluster1, cluster2 = new_cluster1, new_cluster2

            if len(cluster1) < self.threshold or len(cluster2) < self.threshold:

                if len(cluster1) < len(cluster2):
                    small_cluster, large_cluster = cluster1, cluster2
                    small_centroid, large_centroid = centroid1, centroid2
                else:
                    small_cluster, large_cluster = cluster2, cluster1
                    small_centroid, large_centroid = centroid2, centroid1

                small_cluster = small_cluster.tolist()
                large_cluster = large_cluster.tolist()

                points_to_transfer = min(self.threshold, len(large_cluster))
                for _ in range(points_to_transfer):
                    distances = [np.linalg.norm(data[idx] - large_centroid) for idx in large_cluster]
                    farthest_point_idx = np.argmax(distances)
                    point_to_transfer = large_cluster.pop(farthest_point_idx)
                    small_cluster.append(point_to_transfer)

                cluster1, cluster2 = np.array(large_cluster), np.array(small_cluster)
            clusters.append(cluster1.tolist())
            clusters.append(cluster2.tolist())

        return clusters


    def select_clients(self, cluster_id):
        if cluster_id >= len(self.client_clusters) or len(self.client_clusters[cluster_id]) == 0:
            print(f"Cluster {cluster_id} is empty or does not exist. Skipping selection.")
            return []

        if self.random_join_ratio:
            join_clients = \
                np.random.choice(range(self.join_clients, len(self.client_clusters[cluster_id]) + 1), 1, replace=False)[
                    0]
        else:
            join_clients = min(self.join_clients, len(self.client_clusters[cluster_id]))

        selected_clients = list(np.random.choice(self.client_clusters[cluster_id], join_clients, replace=False))
        return selected_clients

    def global_number(self):

        self.cluster_data_sizes = []
        for i, cluster in enumerate(self.client_clusters):
            total_data = sum(client.train_samples for client in cluster)
            print(f"cluster {i} number: {total_data}")
            self.cluster_data_sizes.append(total_data)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            print(f'Client {c.id}: Acc: {ct * 1.0 / ns}, AUC: {auc}')
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def evaluate(self, acc=None):
        stats = self.test_metrics()
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))


        self.last_model_state = copy.deepcopy(self.global_modules.state_dict())
        self.last_accuracy = test_acc

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.cluster_clients(i)
            self.global_number()
            for cluster_id in range(self.num_clusters):
                self.selected_clients = self.select_clients(cluster_id)
                if len(self.selected_clients) == 0:
                    print(f"No clients selected for cluster {cluster_id}. Skipping this round for the cluster.")
                    continue

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}, Cluster: {cluster_id} -------------")
                    print("\nEvaluate before local training")
                    self.evaluate()


                for client in self.selected_clients:
                    client.train_cs_model()
                    client.generate_upload_head()

                self.receive_models(cluster_id)

                if len(self.uploaded_models) == 0:
                    print(f"No models uploaded for cluster {cluster_id}. Skipping aggregation.")
                    continue
                self.aggregate_parameters()
                self.global_head()
                self.global_cs()
            self.aggregate_global_model()

            if self.is_pre_round:
                self.aggregate_global_head()
                self.aggregate_global_cs()
            if not self.is_pre_round:
                self.huffman_aggregate_parameters()
                self.aggregate_merged_models()
                self.send_cluster_ori1()
            self.cluster_aggregated_models = []
            self.cluster_aggregated_heads = []
            self.cluster_aggregated_cs = []

            e_t = time.time()
            self.Budget.append(e_t - s_t)
            print('-' * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def receive_models(self, cluster_id):

        active_train_samples = 0
        for client in self.client_clusters[cluster_id]:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.model.feature_extractor)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_modules.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_normalized_weights(self, w, client_model, q=1):

        server_params = list(self.global_modules.fc1.parameters())

        client_params = list(client_model.fc1.parameters())

        assert len(server_params) == len(client_params), "Parameter count mismatch"

        total_param_diff = 0.0
        for server_param, client_param in zip(server_params, client_params):
            param_diff = (client_param.data - server_param.data).clone()
            total_param_diff += param_diff.norm() ** 2

        total_param_diff = total_param_diff ** 0.5

        weight = w * (total_param_diff ** (q + 1)) / (q + 1)
        return weight

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        weights = []
        total_weight = 0.0
        self.normalized_weights = []
        self.global_modules = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_modules.parameters():
            param.data = torch.zeros_like(param.data)
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        self.cluster_aggregated_models.append(copy.deepcopy(self.global_modules))

    def global_head(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.head_g)

        self.head = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.head.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_head(w, client_model)
        self.cluster_aggregated_heads.append(copy.deepcopy(self.head))


    def add_head(self, w, head):
        for server_param, client_param in zip(self.head.parameters(), head.parameters()):
            server_param.data += client_param.data.clone() * w

    def global_cs(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.gate.cs)

        self.cs = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.cs.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_cs(w, client_model)
        self.cluster_aggregated_cs.append(copy.deepcopy(self.cs))


    def add_cs(self, w, cs):
        for server_param, client_param in zip(self.cs.parameters(), cs.parameters()):
            server_param.data += client_param.data.clone() * w

    def huffman_aggregate_parameters(self):

        self.merged_results = []
        for idx in range(len(self.cluster_aggregated_models)):
            server_params = list(self.global_modules.fc1.parameters())
            model_params = list(self.cluster_aggregated_models[idx].fc1.parameters())
            total_param_diff = 0.0
            for sp, mp in zip(server_params, model_params):
                param_diff = (mp.data - sp.data).clone()
                total_param_diff += param_diff.norm() ** 2
            total_param_diff = total_param_diff ** 0.5
            if total_param_diff == 0:
                trust = 1
            else:
                trust = 1 / total_param_diff

            merged_result = {
                "idx": idx,
                "merged_model": copy.deepcopy(self.cluster_aggregated_models[idx]),
                "merged_head": copy.deepcopy(self.cluster_aggregated_heads[idx]),
                "merged_cs": copy.deepcopy(self.cluster_aggregated_cs[idx]),
                "total_param_diff": total_param_diff,
                "trust": trust,
                "new_diff": total_param_diff
            }
            self.merged_results.append(merged_result)

        total_trust = sum(result['trust'] for result in self.merged_results)
        total_diff = sum(result['total_param_diff'] for result in self.merged_results)
        for result in self.merged_results:
            if result['total_param_diff'] == 0:
                result['trust'] = 1
                result['total_param_diff'] = 0
            else:
                result['trust'] /= total_trust
                result['total_param_diff'] /= total_diff

        sorted_results = sorted(self.merged_results, key=lambda x: x['total_param_diff'])

        sorted_diff = [res['total_param_diff'] for res in sorted_results]
        sorted_trust = [res['trust'] for res in sorted_results]
        cum_model = copy.deepcopy(sorted_results[0]['merged_model'])
        cum_head = copy.deepcopy(sorted_results[0]['merged_head'])
        cum_cs = copy.deepcopy(sorted_results[0]['merged_cs'])
        cumulative_trust = sorted_trust[0]

        send_model = copy.deepcopy(sorted_results[0]['merged_model'])
        send_head = copy.deepcopy(sorted_results[0]['merged_head'])
        send_cs = copy.deepcopy(sorted_results[0]['merged_cs'])


        self.merged_results_cumulative = [{
            "idx": sorted_results[0]['idx'],
            "merged_model": copy.deepcopy(cum_model),
            "merged_head": copy.deepcopy(cum_head),
            "merged_cs": copy.deepcopy(cum_cs),
            "trust_sum": cumulative_trust,
            "total_param_diff": sorted_results[0]['total_param_diff'],
            "new_diff": sorted_results[0]['total_param_diff']
        }]


        for i in range(1, len(sorted_results)):
            T_trust = sorted_trust[i]
            T_diff = sorted_diff[i]
            cumulative_trust += T_trust


            for param_cum, param_new, param_send in zip(cum_model.parameters(), sorted_results[i]['merged_model'].parameters(), send_model.parameters()):

                original_param_cum = param_cum.data.clone()
                param_cum.data = (original_param_cum * (
                            cumulative_trust - T_trust) + T_trust * param_new.data) / cumulative_trust

                param_send.data = (original_param_cum * (
                        cumulative_trust - T_trust) + T_diff * self.k * param_new.data) / (
                                          cumulative_trust - T_trust + T_diff * self.k)

            for param_cum, param_new, param_send in zip(cum_head.parameters(), sorted_results[i]['merged_head'].parameters(), send_head.parameters()):

                original_param_cum = param_cum.data.clone()
                param_cum.data = (original_param_cum * (
                            cumulative_trust - T_trust) + T_trust * param_new.data) / cumulative_trust

                param_send.data = (original_param_cum * (
                        cumulative_trust - T_trust) + T_diff * self.k * param_new.data) / (
                                          cumulative_trust - T_trust + T_diff * self.k)

            for param_cum, param_new, param_send in zip(cum_cs.parameters(), sorted_results[i]['merged_cs'].parameters(), send_cs.parameters()):

                original_param_cum = param_cum.data.clone()
                param_cum.data = (original_param_cum * (
                            cumulative_trust - T_trust) + T_trust * param_new.data) / cumulative_trust

                param_send.data = (original_param_cum * (
                        cumulative_trust - T_trust) + T_diff * self.k * param_new.data) / (
                                          cumulative_trust - T_trust + T_diff * self.k)

            self.merged_results_cumulative.append({
                "idx": sorted_results[i]['idx'],
                "merged_model": copy.deepcopy(send_model),
                "merged_head": copy.deepcopy(send_head),
                "merged_cs": copy.deepcopy(send_cs),
                "trust_sum": cumulative_trust,
                "total_param_diff": sorted_results[i]['total_param_diff'],
                "new_diff": sorted_results[i]['new_diff'],
            })

        for cum_res in self.merged_results_cumulative:
            server_params = list(self.global_modules.fc1.parameters())
            merged_model_params = list(cum_res["merged_model"].fc1.parameters())
            new_diff = 0.0
            for sp, mp in zip(server_params, merged_model_params):
                param_diff = (mp.data - sp.data).clone()
                new_diff += param_diff.norm() ** 2
            new_diff = new_diff ** 0.5
            cum_res['new_diff'] = new_diff
            print(f"Updated Cluster {cum_res['idx']}: New Total Param Diff = {new_diff}")


        for cum_res in self.merged_results_cumulative:
            cluster_idx = cum_res["idx"]
            for client in self.client_clusters[cluster_idx]:
                client.set_parameters_c(cum_res["merged_model"])
                client.set_head_c(cum_res["merged_head"])
                client.set_cs(cum_res["merged_cs"])


    def aggregate_merged_models(self):

        diffs = [result['new_diff'] for result in self.merged_results_cumulative]
        epsilon = 1e-8
        inverse_weights = [1 / diff if diff > epsilon else 1 for diff in diffs]


        cluster_data_sorted = []
        for result in self.merged_results_cumulative:
            cluster_id = result['idx']
            cluster_data_sorted.append(self.cluster_data_sizes[cluster_id])


        end_weights = []
        for i in range(len(cluster_data_sorted)):
            end_weights.append(inverse_weights[i])
        total_weight = sum(end_weights)
        normalized_weights = [weight / total_weight for weight in end_weights]

        if len(normalized_weights) == 1:
            normalized_weights[0] = 1

        self.hf_aggregated_model = copy.deepcopy(self.merged_results_cumulative[0]['merged_model'])
        for param in self.hf_aggregated_model.parameters():
            param.data = torch.zeros_like(param.data)
        for weight, result in zip(normalized_weights, self.merged_results_cumulative):
            model = result['merged_model']
            for agg_param, model_param in zip(self.hf_aggregated_model.parameters(), model.parameters()):
                agg_param.data += weight * model_param.data

        self.hf_aggregated_cs = copy.deepcopy(self.merged_results_cumulative[0]['merged_cs'])
        for param in self.hf_aggregated_cs.parameters():
            param.data = torch.zeros_like(param.data)
        for weight, result in zip(normalized_weights, self.merged_results_cumulative):
            cs = result['merged_cs']
            for agg_param, model_param in zip(self.hf_aggregated_cs.parameters(), cs.parameters()):
                agg_param.data += weight * model_param.data

        self.hf_aggregated_head = copy.deepcopy(self.merged_results_cumulative[0]['merged_head'])
        for param in self.hf_aggregated_head.parameters():
            param.data = torch.zeros_like(param.data)
        for weight, result in zip(normalized_weights, self.merged_results_cumulative):
            head = result['merged_head']
            for agg_param, model_param in zip(self.hf_aggregated_head.parameters(), head.parameters()):
                agg_param.data += weight * model_param.data

        for client in self.clients:
            client.set_parameters(self.hf_aggregated_model)
            client.set_head_g(self.hf_aggregated_head)
            client.set_cs(self.hf_aggregated_cs)



    def send_cluster_ori1(self):
        target_cluster_id = self.merged_results_cumulative[0]["idx"]


        for client in self.client_clusters[target_cluster_id]:
            client.set_parameters_c(self.hf_aggregated_model)
            client.set_head_c(self.hf_aggregated_head)
            client.set_cs(self.hf_aggregated_cs)
        self.merged_results_cumulative = []

    def aggregate_global_model(self):

        if getattr(self, 'hf_aggregated_model', None) is not None:
            self.global_modules = copy.deepcopy(self.hf_aggregated_model)
        else:
            self.global_modules = copy.deepcopy(self.cluster_aggregated_models[0])
            for param in self.global_modules.parameters():
                param.data = torch.zeros_like(param.data)
            cluster_weights = []
            for cluster in self.client_clusters:
                total_data_size = sum(client.train_samples for client in cluster)
                num_clients_in_cluster = len(cluster)
                if num_clients_in_cluster > 0:
                    weight = total_data_size
                    cluster_weights.append(weight)
                else:
                    cluster_weights.append(0)


            total_weight = sum(cluster_weights)
            if total_weight > 0:
                normalized_weights = [weight / total_weight for weight in cluster_weights]
            else:

                normalized_weights = [1.0 / len(cluster_weights)] * len(cluster_weights)

            for cluster_id, cluster_model in enumerate(self.cluster_aggregated_models):
                for global_param, cluster_param in zip(self.global_modules.parameters(), cluster_model.parameters()):
                    global_param.data += cluster_param.data.clone() * normalized_weights[cluster_id]
            if self.is_pre_round:
                self.cluster_aggregated_models = []
                for client in self.clients:
                    client.set_parameters(self.global_modules)
                    client.set_parameters_c(self.global_modules)

    def aggregate_global_head(self):

        assert hasattr(self, 'cluster_aggregated_heads') and len(self.cluster_aggregated_heads) > 0

        self.global_head_model = copy.deepcopy(self.cluster_aggregated_heads[0])
        for param in self.global_head_model.parameters():
            param.data = torch.zeros_like(param.data)

        cluster_weights = []
        for cluster in self.client_clusters:

            total_data_size = sum(client.train_samples for client in cluster)

            num_clients_in_cluster = len(cluster)


            if num_clients_in_cluster > 0:

                weight = total_data_size
                cluster_weights.append(weight)
            else:
                cluster_weights.append(0)


        total_weight = sum(cluster_weights)
        if total_weight > 0:
            normalized_weights = [weight / total_weight for weight in cluster_weights]
        else:

            normalized_weights = [1.0 / len(cluster_weights)] * len(cluster_weights)


        for cluster_id, cluster_head in enumerate(self.cluster_aggregated_heads):
            for global_param, cluster_param in zip(self.global_head_model.parameters(), cluster_head.parameters()):
                global_param.data += cluster_param.data.clone() * normalized_weights[cluster_id]

        self.cluster_aggregated_heads = []

        for client in self.clients:
            client.set_head_g(self.global_head_model)
            client.set_head_c(self.global_head_model)

    def aggregate_global_cs(self):
        assert hasattr(self, 'cluster_aggregated_cs') and len(self.cluster_aggregated_cs) > 0


        self.global_cs_model = copy.deepcopy(self.cluster_aggregated_cs[0])
        for param in self.global_cs_model.parameters():
            param.data = torch.zeros_like(param.data)

        cluster_weights = []
        for cluster in self.client_clusters:

            total_data_size = sum(client.train_samples for client in cluster)

            num_clients_in_cluster = len(cluster)


            if num_clients_in_cluster > 0:

                weight = total_data_size
                cluster_weights.append(weight)
            else:
                cluster_weights.append(0)


        total_weight = sum(cluster_weights)
        if total_weight > 0:
            normalized_weights = [weight / total_weight for weight in cluster_weights]
        else:

            normalized_weights = [1.0 / len(cluster_weights)] * len(cluster_weights)

        for cluster_id, cluster_cs in enumerate(self.cluster_aggregated_cs):
            for global_param, cluster_param in zip(self.global_cs_model.parameters(), cluster_cs.parameters()):
                global_param.data += cluster_param.data.clone() * normalized_weights[cluster_id]

        self.cluster_aggregated_cs = []

        for client in self.clients:
            client.set_cs(self.global_cs_model)


class ConditionalSelection(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(ConditionalSelection, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim * 3),
            nn.LayerNorm([h_dim * 3]),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=8, batch_first=True)

    def forward(self, x, tau=1, hard=False):
        shape = x.shape
        x = self.fc(x)
        x = x.view(shape[0], 3, -1)
        attn_out, _ = self.attn(x, x, x)
        out = F.gumbel_softmax(attn_out, dim=1, tau=tau, hard=hard)
        return out[:, 0, :], out[:, 1, :], out[:, 2, :]

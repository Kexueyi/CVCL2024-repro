import pandas as pd
import torch

class NeuronConceptMapper:
    def __init__(self, csv_path):
        # Load the neuron-to-label mapping from a CSV file
        self.neuron_concept_mapping =self.read_dissect_csv(csv_path)

    def read_dissect_csv(self, csv_path):
        df_concepts = pd.read_csv(csv_path)
        mapping = {}
        for _, row in df_concepts.iterrows():
            layer = row['layer']
            neuron = int(row['unit'])
            concept = row['description']
            similarity = row['similarity']
            
            if layer not in mapping:
                mapping[layer] = {}
            mapping[layer][neuron] = {'concept': concept, 'similarity': similarity}
        return mapping
    
    def get_concepts(self, activations, top_k=2, thres_param=None, mode='per_layer'):
        batch_size = next(iter(activations.values())).shape[0]
        concepts_info = {img_idx: {} for img_idx in range(batch_size)}

        if mode == 'global':
            combined_activations, layer_map, offset_map = self._combine_layers_activations(activations, batch_size)
            activation_mask = self._thresholding(combined_activations, thres_param, top_k, mode)
            self._map_global_indices_to_layers(combined_activations, activation_mask, layer_map, offset_map, concepts_info)
        elif mode == 'per_layer':
            for layer, batch_act in activations.items():
                activation_mask = self._thresholding(batch_act, thres_param, top_k, mode)
                self._get_concepts_per_layer(layer, batch_act, activation_mask, concepts_info)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return concepts_info

    def compose_concepts(self, concepts_info):
        composed_concepts = {}
        for img_idx, layers in concepts_info.items():
            concept_list = []
            for layer, concepts in layers.items():
                for concept in concepts:
                    concept_list.append(concept['concept'])
            composed_concepts[img_idx] = concept_list
        return composed_concepts

    def _combine_layers_activations(self, activations, batch_size):
        combined_activations = []
        layer_map = []
        offset_map = [0]  # offset for each layer
        current_offset = 0  # current offset for the next layer

        for layer, batch_act in activations.items():
            reshaped_activations = batch_act.view(batch_size, -1)
            combined_activations.append(reshaped_activations)
            num_neurons = reshaped_activations.shape[1]
            layer_map.extend([layer] * num_neurons)
            current_offset += num_neurons
            offset_map.append(current_offset)

        combined_activations = torch.cat(combined_activations, dim=1)
        return combined_activations, layer_map, offset_map

    def _thresholding(self, activations, thres_param, top_k, mode):
        if thres_param is not None:
            mean = activations.mean(dim=1, keepdim=True)
            std = activations.std(dim=1, keepdim=True)
            threshold = mean + thres_param * std
            mask = activations > threshold
        else:
            if mode == 'global':
                # Flatten the activations across all layers for each batch item
                flat_activations = activations.view(activations.size(0), -1)
                # print("Flat activations shape:", flat_activations.shape)
                _, indices = torch.topk(flat_activations, k=top_k, largest=True, sorted=True)
                # print("Indices of top k activations:", indices)

                # Create a mask for the top k activations
                mask = torch.zeros_like(flat_activations).bool()
                mask.scatter_(1, indices, True)
                # print("Mask shape:", mask.shape)
                # Reshape the mask to match the original shape of combined activations
                mask = mask.view_as(activations)
                # print("Mask view as:", mask.shape)

            else:
                _, indices = torch.topk(activations, k=top_k, dim=1, largest=True, sorted=True)
                mask = torch.zeros_like(activations).bool()
                mask.scatter_(1, indices, True)
        return mask

    def _map_global_indices_to_layers(self, combined_activations, mask, layer_map, offset_map, concepts_info):
        print("Combined activations shape:", combined_activations.shape)    
        flat_activations = combined_activations.view(-1)
        print("Flat activations shape:", flat_activations.shape)
        active_indices = mask.view(-1).nonzero(as_tuple=True)[0]

        for idx in active_indices:
            global_index = idx.item()
            neuron_index = global_index % combined_activations.shape[1]  # neuron idx
            batch_index = global_index // combined_activations.shape[1]  # batch idx

            # find corresponding layer and local index within the layer
            for i in range(len(offset_map)-1):
                if neuron_index < offset_map[i+1]:
                    corresponding_layer = layer_map[offset_map[i]]
                    local_index = neuron_index - offset_map[i]
                    activation_value = flat_activations[global_index]
                    self._add_concept(corresponding_layer, local_index, activation_value, concepts_info, batch_index)
                    break
            # print("Global index:", global_index)
            # print("Corresponding layer:", corresponding_layer)
            # print("Local index within the layer:", local_index)

    def _get_concepts_per_layer(self, layer, activations, mask, concepts_info):
        batch_size = activations.shape[0]
        for img_idx in range(batch_size):
            active_indices = mask[img_idx].nonzero(as_tuple=True)[0]
            for idx in active_indices:
                unit_index = idx.item()
                self._add_concept(layer, unit_index, activations[img_idx, unit_index], concepts_info, img_idx)

    def _add_concept(self, layer, unit_index, activation_value, concepts_info, img_idx):
        if unit_index in self.neuron_concept_mapping[layer]:
            mapping_info = self.neuron_concept_mapping[layer][unit_index]
            concept_data = {
                # 'layer': layer,
                'unit': unit_index,
                'concept': mapping_info['concept'],
                'similarity': mapping_info['similarity'],
                'activation': activation_value.item()
            }
            if layer not in concepts_info[img_idx]:
                concepts_info[img_idx][layer] = []
            concepts_info[img_idx][layer].append(concept_data)
            
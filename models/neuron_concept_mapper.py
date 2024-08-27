import pandas as pd
import torch

class NeuronConceptMapper:
    def __init__(self, csv_path):
        self.neuron_concept_mapping = self.read_dissect_csv(csv_path)

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

    def get_concepts(self, activations, top_k=2, thres_param=None):
        self.batch_size = next(iter(activations.values())).shape[0]
        if activations[next(iter(activations.keys()))].ndim > 2:
            self.imgs_per_trial = next(iter(activations.values())).shape[1]
            concepts_info = {trial_idx: {img_idx: {} for img_idx in range(self.imgs_per_trial)} 
                            for trial_idx in range(self.batch_size)}
            self._process_trials(activations, top_k, thres_param, concepts_info, self.batch_size, self.imgs_per_trial)
        else:
            concepts_info = {trial_idx: {} for trial_idx in range(self.batch_size)}
            self._process_non_trials(activations, top_k, thres_param, concepts_info, self.batch_size)
        return concepts_info

    def aggregate_concepts(self, concepts_info):
        # Determine the size of the data dynamically
        batch_size = len(concepts_info)
        imgs_per_trial = len(concepts_info[next(iter(concepts_info))])  # Gets the number of images from the first trial
        
        # Initialize the list to store aggregated concept strings
        aggregated_concepts = [[None for _ in range(imgs_per_trial)] for _ in range(batch_size)]
        
        # Iterate over each batch and trial using keys
        for trial_idx, trial_data in concepts_info.items():
            for img_idx, img_data in trial_data.items():
                # Initialize a list to collect unique concepts from all layers
                per_img_concepts = []
                
                # Sort layers to ensure consistent processing order
                sorted_layers = sorted(img_data.keys())
                for layer in sorted_layers:
                    # Extract concepts from the current layer
                    concepts = [info['concept'] for info in img_data[layer]]
                    # Remove duplicates by converting list to a set, then back to a sorted list
                    unique_concepts = sorted(set(concepts))
                    # Join unique concepts from the same layer with '+'
                    per_img_concepts.append(', '.join(unique_concepts))
                
                # Join all layers' unique concepts with ', ' for clear separation
                aggregated_concepts[int(trial_idx)][int(img_idx)] = ', '.join(per_img_concepts)
        
        return aggregated_concepts
    
    def _process_trials(self, activations, top_k, thres_param, concepts_info, batch_size, imgs_per_trial):
        for layer, batch_act in activations.items():
            for trial_idx in range(batch_size):
                for img_idx in range(imgs_per_trial):
                    trial_act = batch_act[trial_idx, img_idx]
                    activation_mask = self._thresholding(trial_act, thres_param, top_k)
                    self._get_concepts_per_layer(layer, trial_act, activation_mask, concepts_info, trial_idx, img_idx)

    def _process_non_trials(self, activations, top_k, thres_param, concepts_info, batch_size):
        for layer, batch_act in activations.items():
            for trial_idx in range(batch_size):
                activation_mask = self._thresholding(batch_act[trial_idx], thres_param, top_k)
                self._get_concepts_per_layer(layer, batch_act[trial_idx], activation_mask, concepts_info, trial_idx)

    def _thresholding(self, activations, thres_param, top_k):
        if thres_param is not None:
            mean = activations.mean(dim=0, keepdim=True)
            std = activations.std(dim=0, keepdim=True)
            threshold = mean + thres_param * std
            mask = activations > threshold
        else:
            _, indices = torch.topk(activations, k=top_k, dim=0, largest=True, sorted=True)
            mask = torch.zeros_like(activations).bool()
            mask.scatter_(0, indices, True)
        return mask

    def _get_concepts_per_layer(self, layer, activations, mask, concepts_info, trial_idx, img_idx=None):
        active_indices = mask.nonzero(as_tuple=True)[0]
        for idx in active_indices:
            unit_index = idx.item()
            activation_value = activations[unit_index]
            self._add_concept(layer, unit_index, activation_value, concepts_info, trial_idx, img_idx)

    def _add_concept(self, layer, unit_index, activation_value, concepts_info, trial_idx, img_idx=None):
        if unit_index in self.neuron_concept_mapping[layer]:
            mapping_info = self.neuron_concept_mapping[layer][unit_index] 
            concept_data = {'unit': unit_index+1,
                            'concept': mapping_info['concept'], 
                            'similarity': mapping_info['similarity'], 
                            'activation': activation_value.item() # Assuming activation_value is a scalar 
                            } 
            if img_idx is not None: # Handle the case with trials 
                if layer not in concepts_info[trial_idx][img_idx]: concepts_info[trial_idx][img_idx][layer] = [] 
                concepts_info[trial_idx][img_idx][layer].append(concept_data) 
            else: # Handle the case without trials 
                if layer not in concepts_info[trial_idx]: concepts_info[trial_idx][layer] = [] 
                concepts_info[trial_idx][layer].append(concept_data)

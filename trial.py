import argparse
from torch.utils.data import DataLoader
import json
import datetime

from data_utils import get_model, get_dataset
from utils import set_seed, save_trial_results
from models.zs_trial_predic import ZSTrialPredic
from models.neuron_concept_mapper import NeuronConceptMapper
from models.feature_extractor import FeatureExtractor
from generate_trials import TrialGenerator

def main(args):
    set_seed(args.seed)
    device = args.device
    
    ## STEP 1: Get Trials (either generate or load)
    if args.trial_path:
        trial_path = args.trial_path
        print(f"Using provided trial: {trial_path}")
    else:
        generator = TrialGenerator(args.data_dir, args.seed)
        trial_path = generator.get_trials()  
        args.trial_path = trial_path
        
    ## STEP 2: Load model, and load dataset from json trial file
    model_name = args.model
    model, transform = get_model(model_name, device)    
    data = get_dataset(dataset_name='object-trial', transform=transform, trials_file=trial_path)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    ## STEP 3: Initialize the Zero-shot trial classifier
    feature_extractor = FeatureExtractor(model_name, model, device)
    classifier = ZSTrialPredic(feature_extractor)
    
    ## STEP 4: Predict the trial results
    if args.map_file and args.top_k is not None:
        mapper = NeuronConceptMapper(args.map_file)
        layers = args.layers.split(",")
        acc, cls_acc, all_concept_info = classifier.predict_with_concepts(dataloader, layers, mapper, args.top_k)
        with open(f'concepts_{args.seed}_{datetime.datetime.now().strftime("%m_%d_%H:%M")}.json', 'w') as json_file:
            json.dump(all_concept_info, json_file, indent=4)
    else: 
        acc, cls_acc = classifier.predict(dataloader)
    
    ## STEP 5: Save the results
    args_dict = vars(args)
    save_trial_results(args_dict, acc, cls_acc)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVCL-TrialZeroShot')
    parser.add_argument('--model', type=str, default='cvcl_res', help='Model name')
    parser.add_argument('--data_dir', type=str, default='/home/Dataset/xueyi/KonkLab/17-objects', help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--layers', type=str, default="vision_encoder.model.layer1,vision_encoder.model.layer2,vision_encoder.model.layer3,vision_encoder.model.layer4",help='List of layers to consider for concept mapping')

    parser.add_argument('--top_k', type=int, help='Top k concepts to consider')
    parser.add_argument('--map_file', type=str, help='Path to the neuron concept mapping file')
    
    parser.add_argument('--trial_path', type=str, help='Optional path to a pre-existing trial JSON file')
    
    args = parser.parse_args()
    main(args)
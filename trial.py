import argparse
from torch.utils.data import DataLoader

from data_utils import get_model, get_dataset
from utils import set_seed, save_trial_results
from models.zs_trial_predic import ZSTrialClassifier

def main(args):
    set_seed(args.seed)
    device = args.device
    
    model_name = args.model
    model, transform = get_model(model_name, device)
    classifier = ZSTrialClassifier(model_name, model, device)
    
    data = get_dataset(dataset_name=args.dataset, 
                       transform=transform, 
                       trials_file=args.trials_file,)
    
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    acc, cls_acc = classifier.predict(dataloader=dataloader)
    
    args_dict = vars(args)
    save_trial_results(args_dict, acc, cls_acc)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVCL-ZeroShot')
    parser.add_argument('--model', type=str, default='cvcl_res', help='Model name')
    parser.add_argument('--dataset', type=str, default='object-trial', help='Dataset name')
    parser.add_argument('--trials_file', type=str, default='datasets/trials/object_trials_42.json', help='Relative Path to json trial file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
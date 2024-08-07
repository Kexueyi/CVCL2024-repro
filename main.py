import argparse
from torch.utils.data import DataLoader
from data_utils import get_model, get_dataset, clean_class_names
from utils import set_seed, ZeroShotClassifier, save_results

def main(args):
    set_seed(args.seed)
    device = args.device
    
    model_name = args.model
    model, preprocess = get_model(model_name, device)
    classifier = ZeroShotClassifier(model_name, model, device)
    
    data = get_dataset(dataset_name=args.dataset, 
                       preprocess=preprocess, 
                       class_file_path=args.class_file, 
                       baby_vocab=args.baby_vocab, 
                       get_attr=args.use_attr, 
                       top_n = args.top_n_desc)
    
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    similarities, predictions, labels, text_combinations = classifier.predict(dataloader=dataloader, prefix=args.prefix, use_attr=args.use_attr)
    
    class_names, clean_cls = data.class_names, data.clean_cls_names
    # print(f"Number of classes: {len(class_names)}")
    
    args_dict = vars(args)
    save_results(args_dict, predictions, labels, similarities, class_names, clean_cls, text_combinations)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVCL-ZeroShot')
    parser.add_argument('--model', type=str, default='cvcl_res', help='Model name')
    parser.add_argument('--dataset', type=str, default='awa2', help='Dataset name')
    parser.add_argument('--class_file', type=str, default='classes.txt', help='Relative Path to class file under dataset folder')
    parser.add_argument('--baby_vocab', action='store_true', default=False, help='Use baby_vocab or not')
    parser.add_argument('--use_attr', action='store_true', default=False, help='Use attributes or not')
    parser.add_argument('--top_n_desc', type=int, default=5, help='Number of top descriptions to use')
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help="""\
    Prefix for class names. This modifies the input text to fit the model's requirements.
    Example:
    --prefix "a photo of a " (for a CLIP model, prepends the phrase to class names)
    """
    )
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
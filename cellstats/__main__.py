import argparse
import os
import cellstats.models as models
import cellstats.post_processing as post_processing

def main():
    parser = argparse.ArgumentParser(description="Microscopy images processing.")
    subparsers = parser.add_subparsers(help="Functions", dest="command")

    
    parser_predict = subparsers.add_parser("predict", help="Predict features from an image or batch."\
        f" Currently available features are {post_processing.FeatureExtractor.ALL_FEATURES}")
    parser_predict.add_argument('input_file', help='Input file or directory', type=str)
    parser_predict.add_argument('output_file', help='Path to output file which will contain features', type=str)
    parser_predict.add_argument('--model', help='Name of model form .cellstats folder or path to one', type=str)
    parser_predict.add_argument("--gpu", required=False, action="store_true",
                            help="Use gpu acceleration if available")
    parser_predict.add_argument('--channel', required=False, help='Channel to segment, default is 2 for green.',
                                    type=int, default=2)
    parser_predict.add_argument('--channel2', required=False, help='Second channel to segment, default is 0 for None.',
                                    type=int, default=0)
    parser_predict.add_argument("--features", required=False, nargs='+', default=None,
                                help="List of features to predict. By default predicts all available features.")
            


    
    parser_model = subparsers.add_parser("model", help="manage cellpose models")
    subparsers_model = parser_model.add_subparsers(help="Functions", dest="model_command")

    parser_model_add = subparsers_model.add_parser("add", help="Add a model to the hidden .cellstats folder")
    parser_model_add.add_argument('model_file', help="The model file to copy", type=str)
    parser_model_add.add_argument('--name', required=False, type=str, help="The name the model will be stored with.")
    parser_model_add.add_argument("--overwrite", required=False, action="store_true",
                            help="Overwrite existing model with the same name, if exists")

    parser_model_rename = subparsers_model.add_parser("rename", help="Rename a model in the hidden .cellstats folder")
    parser_model_rename.add_argument("old_name", help="The model to rename", type=str)
    parser_model_rename.add_argument("new_name", help="The new name", type=str)
    parser_model_rename.add_argument("--overwrite", required=False, action="store_true",
                            help="Overwrite existing model with the same name, if exists")
    

    parser_model_remove = subparsers_model.add_parser("remove", help="Remove a model from the hidden .cellstats folder")
    parser_model_remove.add_argument("name", help="The model to remove", type=str)

    args = parser.parse_args()
    

    if args.command == "model":
        if args.model_command == "add":
            models.add_model(args.model_file, args.name, args.overwrite)
        elif args.model_command == "rename":
            models.rename_model(args.old_name, args.new_name, args.overwrite)
        elif args.model_command == "remove":
            models.rename_model(args.name)

    elif args.command == "predict":
        masks = models.predict_masks(args.input_file, args.model, args.gpu, [[args.channel, args.channel2]])
        fe = post_processing.FeatureExtractor(masks, 1) # Placeholder for scale detection
        df = fe.get_features(args.features)
        df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
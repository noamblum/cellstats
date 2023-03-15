import argparse
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    parser = argparse.ArgumentParser(description="Microscopy images processing.")
    subparsers = parser.add_subparsers(help="Functions", dest="command")

    
    parser_predict = subparsers.add_parser("predict", help="Predict features from an image or batch.")
    parser_predict.add_argument('input_file', help='Input file or directory', type=str)
    parser_predict.add_argument('output_file', help='Path to output file which will contain features', type=str)
    parser_predict.add_argument('--model', help='Name of model form .cellstats folder or path to one', type=str)
    parser_predict.add_argument('--channel', required=False, help='Channel to segment. If using CZI files, can be a'\
                                ' string specifying channel name or an integer specifying channel index. '
                                'For regular image files: 0 - grayscale, 1 - red, 2 - green, 3 - blue.'
                                ' In all cases default is 0.', default=0)
    parser_predict.add_argument("--gpu", required=False, action="store_true",
                            help="Use gpu acceleration if available")
    parser_predict.add_argument("--features", required=False, nargs='+', default=None,
                                help="List of features to predict. By default predicts all available features.")
    parser_predict.add_argument("--save_outlines", required=False, action="store_true",
                            help="Save the predicted masks in a folder alongside the output file.")
    parser_predict.add_argument("-v" ,"--verbose", required=False, action="store_true",
                            help="Verbose output")
            


    
    parser_model = subparsers.add_parser("model", help="manage cellpose models")
    subparsers_model = parser_model.add_subparsers(help="Functions", dest="model_command")

    parser_model_add = subparsers_model.add_parser("add", help="Add a model to the hidden .cellstats folder")
    parser_model_add.add_argument('model_file', help="The model file to copy", type=str)
    parser_model_add.add_argument('-n', '--name', required=False, type=str, help="The name the model will be stored with.")
    parser_model_add.add_argument("--overwrite", required=False, action="store_true",
                            help="Overwrite existing model with the same name, if exists")
    parser_model_add.add_argument("-e", "--env", "--environment", required=False, action="store_true",
                            help="Use the environment-wide model repository, if initialized")

    parser_model_rename = subparsers_model.add_parser("rename", help="Rename a model in the hidden .cellstats folder")
    parser_model_rename.add_argument("old_name", help="The model to rename", type=str)
    parser_model_rename.add_argument("new_name", help="The new name", type=str)
    parser_model_rename.add_argument("--overwrite", required=False, action="store_true",
                            help="Overwrite existing model with the same name, if exists")
    parser_model_rename.add_argument("-e", "--env", "--environment", required=False, action="store_true",
                            help="Use the environment-wide model repository, if initialized")
    

    parser_model_remove = subparsers_model.add_parser("remove", help="Remove a model from the hidden .cellstats folder")
    parser_model_remove.add_argument("name", help="The model to remove", type=str)
    parser_model_remove.add_argument("-e", "--env", "--environment", required=False, action="store_true",
                            help="Use the environment-wide model repository, if initialized")


    parser_model_ls = subparsers_model.add_parser("ls", help="List available models")

    args = parser.parse_args()
    

    if args.command == "model":
        from cellstats import models
        if args.model_command == "add":
            models.add_model(args.model_file, args.name, args.overwrite, args.env)
        elif args.model_command == "rename":
            models.rename_model(args.old_name, args.new_name, args.overwrite, args.env)
        elif args.model_command == "remove":
            models.remove_model(args.name, args.env)
            
        elif args.model_command == "ls":
            mdls = models.list_models()
            separator = '\n  - '
            env_models = separator.join(mdls["environment"])
            local_models = separator.join(mdls["local"])

            if not models.environment_repository_initialized():
                env_models = f"\n  {bcolors.WARNING}Uninitialized{bcolors.ENDC}"
            elif env_models != '':
                env_models = separator + env_models
            
            if local_models:
                local_models = separator + local_models
            print(f"Environment:{env_models}\nLocal:{local_models}")

    elif args.command == "predict":
        from cellstats import models
        from cellstats import post_processing
        from cellstats import io
        if args.save_outlines:
            outdir = os.path.join(os.path.dirname(os.path.abspath(args.output_file)), "segmentation_outlines")
        else:
            outdir = None
        channel = int(args.channel) if args.channel.isdigit() else args.channel
        masks, image_names = models.predict_masks(args.input_file, args.model, args.gpu, channel,
                                    args.verbose, None, return_image_names=True)
        if outdir is not None:
            io.save_image_outlines(args.input_file, outdir, masks, channel, verbose=True)
        fe = post_processing.FeatureExtractor(masks, files=image_names, scales=io.load_image_scales(args.input_file))
        df = fe.get_features(args.features)
        df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
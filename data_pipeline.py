import argparse
import yaml
from apulu.data_pipeline import DataPipeline


def main(args):
    ## data phase
    configs = yaml.safe_load(open(args.config_path))
    data_pipeline = DataPipeline(
        config_path=args.config_path,
        components=configs["data_pipeline_config"]["components"],
    )
    data_pipeline.run_pipeline(redownload=configs["data_pipeline_config"]["redownload"])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="entrypoint for data pipeline")
    parser.add_argument(
        "--config_path", type=str, help="the path of configuration file"
    )
    args = parser.parse_args()
    main(args)
